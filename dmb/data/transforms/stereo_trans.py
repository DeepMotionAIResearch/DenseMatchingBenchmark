import random
import numbers
import numpy as np

import torch
from torch.nn.functional import pad
import torchvision.transforms.functional as F


class ToTensor(object):
    """
    convert numpy.ndarray to torch.floatTensor, in [Channels, Height, Width]
    """
    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k].copy())
        return sample


class CenterCrop(object):
    """Crops the given image at central location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
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


class Normalize(object):
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


class StereoPad(object):
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

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0

        sample['leftImage'] = pad(
            sample['leftImage'], [pad_left, pad_right, pad_top, pad_bottom],
            mode='constant', value=0
        )
        sample['rightImage'] = pad(
            sample['rightImage'], [pad_left, pad_right, pad_top, pad_bottom],
            mode='constant', value=0
        )

        return sample
