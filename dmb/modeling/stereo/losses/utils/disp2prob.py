import warnings

import torch
import torch.nn.functional as F


def isNaN(x):
    return x != x


class Disp2Prob(object):
    """
    Convert disparity map to matching probability volume

    Args:
        gtDisp (Tensor): ground truth disparity map, in [BatchSize, 1, Height, Width] layout
        max_disp (int): the maximum of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (optional, int): the step between near disparity index
        disp_sample (optional, Tensor):
            if not None, direct provide the disparity samples for each pixel in [BatchSize, disp_sample_number, Height, Width] layout

    Outputs:
        ground truth probability volume (Tensor): in [BatchSize, disp_sample_number, Height, Width] layout

    """

    def __init__(self, gtDisp, max_disp, start_disp=0, dilation=1, disp_sample=None):

        if not isinstance(max_disp, int):
            raise TypeError('int is expected, got {}'.format(type(max_disp)))

        if not torch.is_tensor(gtDisp):
            raise TypeError('torch.Tensor is expected, got {}'.format(type(gtDisp)))

        if not isinstance(start_disp, int):
            raise TypeError('int is expected, got {}'.format(type(start_disp)))

        if not isinstance(dilation, int):
            raise TypeError('int is expected, got {}'.format(type(dilation)))


        #  B x 1 x H x W
        assert gtDisp.size(1) == 1, '2nd dimension size should be 1, got {}'.format(gtDisp.size(1))

        if disp_sample is not None:
            if not isinstance(disp_sample, torch.Tensor):
                raise TypeError("torch.Tensor expected, but got {}".format(type(disp_sample)))

            disp_sample = disp_sample.to(gtDisp.device)

            idb, idc, idh, idw = disp_sample.shape
            gtb, gtc, gth, gtw = gtDisp.shape

            assert (idb, idh, idw) == (gtb, gth, gtw), 'The (B, H, W) should be same between ' \
                                                       'ground truth disparity map and disparity index!'

        self.gtDisp = gtDisp
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.end_disp = start_disp + max_disp - 1
        self.dilation = dilation
        self.disp_sample = disp_sample
        self.eps = 1e-40

    def getCost(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        if self.disp_sample is None:
            self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation

            # [disp_sample_number]
            self.disp_sample = torch.linspace(
                self.start_disp, self.end_disp, self.disp_sample_number
            ).to(self.gtDisp.device)

            # [BatchSize, disp_sample_number, Height, Width]
            self.disp_sample = self.disp_sample.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()


        # value of gtDisp must within (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        # [BatchSize, disp_sample_number, Height, Width]
        cost = self.calCost()

        # let the outliers' cost to be -inf
        # [BatchSize, disp_sample_number, Height, Width]
        cost = cost * mask - 1e12

        # in case cost is NaN
        if isNaN(cost.min()) or isNaN(cost.max()):
            print('Cost ==> min: {:.4f}, max: {:.4f}'.format(cost.min(), cost.max()))
            print('Disparity Sample ==> min: {:.4f}, max: {:.4f}'.format(self.disp_sample.min(),
                                                                         self.disp_sample.max()))
            print('Disparity Ground Truth after mask out ==> min: {:.4f}, max: {:.4f}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'cost contains NaN!")

        return cost

    def getProb(self):
        # [BatchSize, 1, Height, Width]
        b, c, h, w = self.gtDisp.shape
        assert c == 1

        # if start_disp = 0, dilation = 1, then generate disparity candidates as [0, 1, 2, ... , maxDisp-1]
        if self.disp_sample is None:
            self.disp_sample_number = (self.max_disp + self.dilation - 1) // self.dilation

            # [disp_sample_number]
            self.disp_sample = torch.linspace(
                self.start_disp, self.end_disp, self.disp_sample_number
            ).to(self.gtDisp.device)

            # [BatchSize, disp_sample_number, Height, Width]
            self.disp_sample = self.disp_sample.repeat(b, h, w, 1).permute(0, 3, 1, 2).contiguous()


        # value of gtDisp must within (start_disp, end_disp), otherwise, we have to mask it out
        mask = (self.gtDisp > self.start_disp) & (self.gtDisp < self.end_disp)
        mask = mask.detach().type_as(self.gtDisp)
        self.gtDisp = self.gtDisp * mask

        # [BatchSize, disp_sample_number, Height, Width]
        probability = self.calProb()

        # let the outliers' probability to be 0
        # in case divide or log 0, we plus a tiny constant value
        # [BatchSize, disp_sample_number, Height, Width]
        probability = probability * mask + self.eps

        # in case probability is NaN
        if isNaN(probability.min()) or isNaN(probability.max()):
            print('Probability ==> min: {:.4f}, max: {:.4f}'.format(probability.min(), probability.max()))
            print('Disparity Sample ==> min: {:.4f}, max: {:.4f}'.format(self.disp_sample.min(),
                                                                         self.disp_sample.max()))
            print('Disparity Ground Truth after mask out ==> min: {:.4f}, max: {:.4f}'.format(self.gtDisp.min(),
                                                                                      self.gtDisp.max()))
            raise ValueError(" \'probability contains NaN!")

        return probability


    def calProb(self):
        raise NotImplementedError

    def calCost(self):
        raise NotImplementedError


class LaplaceDisp2Prob(Disp2Prob):
    # variance is the diversity of the Laplace distribution
    def __init__(self, gtDisp, max_disp, variance=1, start_disp=0, dilation=1, disp_sample=None):
        super(LaplaceDisp2Prob, self).__init__(gtDisp, max_disp, start_disp, dilation, disp_sample)
        self.variance = variance

    def calCost(self):
        # 1/N * exp( - (d - d{gt}) / var), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        cost = ((-torch.abs(self.disp_sample - self.gtDisp)) / self.variance)

        return cost

    def calProb(self):
        cost = self.calCost()
        probability = F.softmax(cost, dim=1)

        return probability


class GaussianDisp2Prob(Disp2Prob):
    # variance is the variance of the Gaussian distribution
    def __init__(self, gtDisp, max_disp, variance=1, start_disp=0, dilation=1, disp_sample=None):
        super(GaussianDisp2Prob, self).__init__(gtDisp, max_disp, start_disp, dilation, disp_sample)
        self.variance = variance

    def calCost(self):
        # 1/N * exp( - (d - d{gt})^2 / b), N is normalization factor, [BatchSize, maxDisp, Height, Width]
        distance = (torch.abs(self.disp_sample - self.gtDisp))
        cost = (- distance.pow(2.0) / self.variance)

        return cost

    def calProb(self):
        cost = self.calCost()
        probability = F.softmax(cost, dim=1)

        return probability


class OneHotDisp2Prob(Disp2Prob):
    # variance is the variance of the OneHot distribution
    def __init__(self, gtDisp, max_disp, variance=1, start_disp=0, dilation=1, disp_sample=None):
        super(OneHotDisp2Prob, self).__init__(gtDisp, max_disp, start_disp, dilation, disp_sample)
        self.variance = variance

    def getProb(self):
        # |d - d{gt}| < variance, [BatchSize, maxDisp, Height, Width]
        probability = torch.lt(torch.abs(self.disp_sample - self.gtDisp), self.variance).type_as(self.gtDisp)

        return probability
