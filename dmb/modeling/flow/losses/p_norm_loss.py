import torch
import torch.nn as nn
import torch.nn.functional as F

def sparse_max_pool(input, size):
    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output


def zero_mask(input, eps=1e-12):
    mask = abs(input) < eps
    return mask


class PNormLoss(object):
    """
    Norm_P loss
    Args:
        p, (float): p norm
        epsilon, (float): a small const value which balance the absolute difference
                          between estimated and ground truth flow
        weights (list of float or None): weight for each scale of estFlow.
        sparse (bool): whether the ground-truth disparity is sparse,
            for example, KITTI is sparse, but SceneFlow is not, default is False.
    Inputs:
        estFlow (Tensor or list of Tensor): the estimated flow map,
            in [BatchSize, 2, Height, Width] layout.
        gtFlow (Tensor): the ground truth flow map,
            in [BatchSize, 2, Height, Width] layout.
    Outputs:
        loss (dict), the loss of each level
    """

    def __init__(self, p=2, epsilon=0, weights=None, sparse=False):
        self.p = p
        self.epsilon = epsilon
        self.weights = weights
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = sparse_max_pool
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estFlow, gtFlow):
        B, C, H, W = estFlow.shape
        scaled_gtFlow = gtFlow
        if gtFlow.shape[-2] != H or gtFlow.shape[-1] != W:
            # compute scale per level and scale gtFlow
            scale = gtFlow.shape[-1] / (W * 1.0)
            scaled_gtFlow = gtFlow / scale
            scaled_gtFlow = self.scale_func(scaled_gtFlow, (H, W))

        # calculate loss
        # [B, 2, H, W]
        diff = torch.abs(scaled_gtFlow - estFlow) + self.epsilon

        # [B, H, W]
        loss = torch.norm(diff, p=self.p, dim=1, keepdim=False)

        # get invalid mask where motion is invalid
        # [B, H, W]
        gt_u, gt_v = scaled_gtFlow[:, 0, :, :], scaled_gtFlow[:, 1, :, :]
        # [B, H, W]
        invalid_mask = torch.isnan(gt_u) | torch.isnan(gt_v)
        # in a image with shape [H, W], the maximum motion is within [-W, W] and [-H, H]
        invalid_mask = invalid_mask | ((gt_u > W) | (gt_u < -W))
        invalid_mask = invalid_mask | ((gt_v > H) | (gt_v < -H))

        if self.sparse:
            # mask for valid disparity
            # Attention: the invalid flow of KITTI is set as 0, be sure to mask it out
            # [B, H, W]
            sparse_mask = (zero_mask(gt_u)) & (zero_mask(gt_v))
            invalid_mask = invalid_mask | sparse_mask

        loss = loss[~invalid_mask]
        loss = loss.sum() / B

        return loss

    def __call__(self, estFlow, gtFlow):
        if not isinstance(estFlow, (list, tuple)):
            estFlow = [estFlow]

        if self.weights is None:
            self.weights = [1.0] * len(estFlow)

        # compute loss for per level
        loss_all_level = []
        for est_flow_per_lvl in estFlow:
            loss_all_level.append(
                self.loss_per_level(est_flow_per_lvl, gtFlow)
            )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "p_norm_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Loss weights: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Flow is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'PNormLoss'
