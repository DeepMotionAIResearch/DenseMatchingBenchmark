from __future__ import division
import torch
import torch.nn.functional
import random
import numpy as np
import numbers
import torchvision.transforms.functional as F
import scipy.ndimage as ndimage


#************************************* Only Numpy Allowed ********************************* #
class RandomRotate(object):
    """
    Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bi-linear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flow map will be off.
    All in [Channels, Height, Width] layout, can be type numpy.ndarray, torch.Tensor
    Inputs:
        sample, (dict):
            leftImage, (numpy.ndarray): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray): in [Channels, Height, Width] layout
            flow, (numpy.ndarray): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (numpy.ndarray): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray): in [Channels, Height, Width] layout
            flow, (numpy.ndarray): in [2, Height, Width] layout
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, sample):
        if 'flow' not in sample.keys():
            return sample

        applied_angle = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2
        angle2 = applied_angle + diff/2
        angle1_rad = angle1*np.pi/180

        h, w = sample['leftImage'].shape[-2:]

        def rotate_flow(k, i, j):
            return -k*(j-w/2)*(diff*np.pi/180) + (1-k)*(i-h/2)*(diff*np.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, sample['flow'].shape)
        sample['flow'] += rotate_flow_map

        sample['leftImage'] = ndimage.interpolation.rotate(sample['leftImage'], angle1, axes=(-2, -1), reshape=self.reshape, order=self.order)
        sample['rightImage'] = ndimage.interpolation.rotate(sample['rightImage'], angle2, axes=(-2, -1), reshape=self.reshape, order=self.order)
        sample['flow'] = ndimage.interpolation.rotate(sample['flow'], angle1, axes=(-2, -1), reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down, clockwise
        flow = np.copy(sample['flow'])
        sample['flow'][0,:,:] = np.cos(angle1_rad)*flow[0,:,:] + np.sin(angle1_rad)*flow[1,:,:]
        sample['flow'][1,:,:] = -np.sin(angle1_rad)*flow[0,:,:] + np.cos(angle1_rad)*flow[1,:,:]

        return sample


class ToTensor(object):
    """
    Inputs:
        sample, (dict):
            leftImage, (numpy.ndarray): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray): in [Channels, Height, Width] layout
            flow, (numpy.ndarray): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (numpy.ndarray): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray): in [Channels, Height, Width] layout
            flow, (numpy.ndarray): in [2, Height, Width] layout
    """
    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k].copy())
        return sample


#************************************* Both Tensor, Numpy Allowed ********************************* #

class CenterCrop(object):
    """Crops the given image at central location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Inputs:
        sample, (dict):
            leftImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            flow, (numpy.ndarray, tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            flow, (numpy.ndarray, tensor): in [2, Height, Width] layout
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        h, w = sample['leftImage'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = (w - tw) // 2
        y1 = (h - th) // 2

        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = sample[k][:, y1: y1 + th, x1: x1 + tw]
        return sample


class RandomCrop(object):
    """Crops the given image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Inputs:
        sample, (dict):
            leftImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            flow, (numpy.ndarray, tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            flow, (numpy.ndarray, tensor): in [2, Height, Width] layout
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        h, w = sample['leftImage'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = sample[k][:, y1: y1 + th, x1: x1 + tw]
        return sample


class RandomTranslate(object):
    """
    Inputs:
        sample, (dict):
            leftImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            flow, (numpy.ndarray, tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            rightImage, (numpy.ndarray, tensor): in [Channels, Height, Width] layout
            flow, (numpy.ndarray, tensor): in [2, Height, Width] layout
    """
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, sample):
        h, w = sample['leftImage'].shape[-2:]
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return sample
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        sample['leftImage'] = sample['leftImage'][:, y1:y2, x1:x2]
        sample['rightImage'] = sample['rightImage'][:, y3:y4, x3:x4]

        if sample['flow'] is not None and isinstance(sample['flow'], (np.ndarray, torch.Tensor)):
            sample['flow'][0, :, :] += tw
            sample['flow'][1, :, :] += th

        return sample


#************************************* Only Tensor Allowed ********************************* #

class Normalize(object):
    """
    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, pad_Height, pad_Width] layout
            rightImage, (tensor): in [Channels, pad_Height, pad_Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['leftImage'] = F.normalize(
            sample['leftImage'], mean=self.mean, std=self.std
        )
        sample['rightImage'] = F.normalize(
            sample['rightImage'], mean=self.mean, std=self.std
        )
        return sample


class CenterCat(object):
    """
    Zero padding left and right image at top, bottom, left, and right.
    Make the high and width of array fit the divisor.
    Args:
        divisor, (int): make the height and width can be divided by divisor
    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, pad_Height, pad_Width] layout
            rightImage, (tensor): in [Channels, pad_Height, pad_Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['leftImage'].shape[-2:]
        H, W = self.size
        pad_top = (H - h) // 2
        pad_bottom = (H - h) - pad_top
        pad_left = (W - w) // 2
        pad_right = (W - w) - pad_left

        pad = [pad_top, pad_bottom, pad_left, pad_right]

        sample['leftImage'] = torch.nn.functional.pad(sample['leftImage'], pad=pad, mode='constant', value=0)
        sample['rightImage'] = torch.nn.functional.pad(sample['rightImage'], pad=pad, mode='constant', value=0)

        return sample


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given torch.floatTensor with a probability of 0.5
    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __call__(self, sample):
        if random.random() < 0.5:
            for k in sample.keys():
                if sample[k] is not None and isinstance(sample[k], (torch.Tensor)):
                    sample[k] = torch.flip(sample[k], [-1])
                    # if flow flipped
                    if 'flow' in k:
                        sample[k][:, :, 0] *= -1

        return sample


class RandomVerticalFlip(object):
    """
    Randomly horizontally flips the given torch.floatTensor with a probability of 0.5
    All in [Channels, Height, Width] layout, can be type torch.Tensor
    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __call__(self, sample):
        if random.random() < 0.5:
            for k in sample.keys():
                if sample[k] is not None and isinstance(sample[k], (torch.Tensor)):
                    sample[k] = torch.flip(sample[k], [-2])
                    # if flow flipped
                    if 'flow' in k:
                        sample[k][:, :, 1] *= -1

        return sample


#**************************** Only Tensor Allowed (Color Augmentation)************************ #


class Grayscale(object):

    def grayscale_img(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs

    def __call__(self, sample):
        leftImage = self.grayscale_img(sample['leftImage'])
        rightImage = self.grayscale_img(sample['rightImage'])
        return leftImage, rightImage


class Saturation(object):
    """

    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        leftImage, rightImage = Grayscale()(sample)
        alpha = random.uniform(0, self.var)
        sample['leftImage'] = sample['leftImage'].lerp(leftImage, alpha)
        sample['rightImage'] = sample['rightImage'].lerp(rightImage, alpha)

        return sample


class Brightness(object):
    """

    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        leftImage = sample['leftImage'].new().resize_as_(sample['leftImage']).zero_()
        rightImage = sample['rightImage'].new().resize_as_(sample['rightImage']).zero_()
        alpha = random.uniform(0, self.var)

        sample['leftImage'] = sample['leftImage'].lerp(leftImage, alpha)
        sample['rightImage'] = sample['rightImage'].lerp(rightImage, alpha)

        return sample


class Contrast(object):
    """

    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, sample):
        leftImage, rightImage = Grayscale()(sample)
        leftImage = leftImage.fill_(leftImage.mean())
        rightImage = rightImage.fill_(rightImage.mean())
        alpha = random.uniform(0, self.var)

        sample['leftImage'] = sample['leftImage'].lerp(leftImage, alpha)
        sample['rightImage'] = sample['rightImage'].lerp(rightImage, alpha)

        return sample


class RandomOrder(object):
    """
    Composes several transforms together in random order.

    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        if self.transforms is None:
            return sample
        order = torch.randperm(len(self.transforms))
        for i in order:
            sample = self.transforms[i](sample)
        return sample


class ColorJitter(RandomOrder):
    """
    Color augmentation.
    Inputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    Outputs:
        sample, (dict):
            leftImage, (tensor): in [Channels, Height, Width] layout
            rightImage, (tensor): in [Channels, Height, Width] layout
            flow, (tensor): in [2, Height, Width] layout
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        transforms = []
        if brightness != 0:
            transforms.append(Brightness(brightness))
        if contrast != 0:
            transforms.append(Contrast(contrast))
        if saturation != 0:
            transforms.append(Saturation(saturation))

        super(ColorJitter, self).__init__(transforms)
