import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dmb.modeling.stereo.losses.utils import LaplaceDisp2Prob, GaussianDisp2Prob, OneHotDisp2Prob


class StereoFocalLoss(object):
    """
    Under the same start disparity and maximum disparity, calculating all estimated cost volumes' loss.
        Args:
            max_disp (int): the max of Disparity. default: 192
            start_disp (int): the start searching disparity index, usually be 0
            dilation (optional, int): the step between near disparity index,
                it mainly used in gt probability volume generation
            weights (list of float or None): loss weight for each scale of estCost.
            focal_coefficient (float): stereo focal loss focal coefficient,
                details please refer to paper. default: 0.0
            sparse (bool): whether the ground-truth disparity is sparse,
                for example, KITTI is sparse, but SceneFlow is not. default: False

        Inputs:
            estCost (Tensor or list of Tensor): the estimated cost volume,
                in [BatchSize, disp_sample_number, Height, Width] layout,
                the disp_sample_number can be: (max_disp + dilation - 1) / dilation or disp_index.shape[1]
            gtDisp (Tensor): the ground truth disparity map,
                in [BatchSize, 1, Height, Width] layout.
            variance (int, Tensor or list of Tensor): the variance of distribution,
                details please refer to paper, in [BatchSize, 1, Height, Width] layout.
            disp_sample (optional, (Tensor or list of Tensor)):
                if not None, direct provide the disparity samples for each pixel
                in [BatchSize, disp_sample_number, Height, Width] layout

        Outputs:
            weighted_loss_all_level (Tensor), the weighted loss of all levels

        Note:
            Before calculate loss, the estCost shouldn't be normalized,
              because we will use softmax for normalization
    """

    def __init__(
            self, max_disp, start_disp=0,
            dilation=1, weights=None,
            focal_coefficient=0.0,
            sparse=False
    ):
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.end_disp = self.max_disp + self.start_disp - 1
        self.dilation = dilation
        self.weights = weights
        self.focal_coefficient = focal_coefficient
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estCost, gtDisp, variance, dilation, disp_sample):
        B, C, H, W = estCost.shape
        scaled_gtDisp = gtDisp.clone()
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale factor for per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp.clone() / scale

            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))

        # mask for valid disparity
        # (start_disp, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        lower_bound = self.start_disp
        upper_bound = lower_bound + int(self.max_disp / scale)
        mask = (scaled_gtDisp > lower_bound) & (scaled_gtDisp < upper_bound)
        mask = mask.detach_().type_as(scaled_gtDisp)
        if mask.sum() < 1.0:
            print('Stereo focal loss: there is no point\'s '
                  'disparity is within [{},{})!'.format(lower_bound, upper_bound))
            scaled_gtProb = torch.zeros_like(estCost)  # let this sample have loss with 0
        else:
            # transfer disparity map to probability map
            mask_scaled_gtDisp = scaled_gtDisp * mask
            scaled_gtProb = LaplaceDisp2Prob(
                mask_scaled_gtDisp, max_disp=int(self.max_disp / scale), variance=variance,
                start_disp=self.start_disp, dilation=dilation, disp_sample=disp_sample
            ).getProb()

        # stereo focal loss
        valid_pixel_number = mask.float().sum()
        if valid_pixel_number < 1.0:
            valid_pixel_number = 1.0
        estLogProb = F.log_softmax(estCost, dim=1)
        weight = (1.0 - scaled_gtProb).pow(-self.focal_coefficient).type_as(scaled_gtProb)
        loss = -((scaled_gtProb * estLogProb) * weight * mask.float()).sum() / valid_pixel_number

        return loss

    def __call__(self, estCost, gtDisp, variance, disp_sample=None):
        if not isinstance(estCost, (list, tuple)):
            estCost = [estCost]

        if self.weights is None:
            self.weights = 1.0

        if not isinstance(self.weights, (list, tuple)):
            self.weights = [self.weights] * len(estCost)

        if not isinstance(self.dilation, (list, tuple)):
            self.dilation = [self.dilation] * len(estCost)

        if not isinstance(variance, (list, tuple)):
            variance = [variance] * len(estCost)

        if disp_sample is None:
            disp_sample = [disp_sample] * len(estCost)
        else:
            if not isinstance(disp_sample, (list, tuple)):
                # Use same disparity samples for each estimated cost volume
                disp_sample = [disp_sample] * len(estCost)

        # compute loss for per level
        loss_all_level = []
        for est_cost_per_lvl, var, dt, ds in zip(estCost, variance, self.dilation, disp_sample):
            loss_all_level.append(
                self.loss_per_level(est_cost_per_lvl, gtDisp, var, dt, ds))

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "stereo_focal_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Focal coefficient: {}\n'.format(self.focal_coefficient)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'StereoFocalLoss'
