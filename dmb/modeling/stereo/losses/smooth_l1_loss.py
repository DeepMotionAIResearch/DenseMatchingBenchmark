import torch
import torch.nn as nn
import torch.nn.functional as F


class DispSmoothL1Loss(object):
    """
    Args:
        max_disp (int): the max of Disparity. default is 192
        start_disp (int): the start searching disparity index, usually be 0
        weights (list of float or None): weight for each scale of estCost.
        sparse (bool): whether the ground-truth disparity is sparse,
            for example, KITTI is sparse, but SceneFlow is not, default is False.
    Inputs:
        estDisp (Tensor or list of Tensor): the estimated disparity map,
            in [BatchSize, 1, Height, Width] layout.
        gtDisp (Tensor): the ground truth disparity map,
            in [BatchSize, 1, Height, Width] layout.
    Outputs:
        loss (dict), the loss of each level
    """

    def __init__(self, max_disp, start_disp=0, weights=None, sparse=False):
        self.max_disp = max_disp
        self.weights = weights
        self.start_disp = start_disp
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estDisp, gtDisp):
        N, C, H, W = estDisp.shape
        scaled_gtDisp = gtDisp
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp / scale
            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))

        # mask for valid disparity
        # (start disparity, max disparity / scale)
        # Attention: the invalid disparity of KITTI is set as 0, be sure to mask it out
        mask = (scaled_gtDisp > self.start_disp) & (scaled_gtDisp < (self.max_disp / scale))
        if mask.sum() < 1.0:
            print('SmoothL1 loss: there is no point\'s disparity is in ({},{})!'.format(self.start_disp,
                                                                                        self.max_disp / scale))
            loss = (torch.abs(estDisp - scaled_gtDisp) * mask.float()).mean()
            return loss

        # smooth l1 loss
        loss = F.smooth_l1_loss(estDisp[mask], scaled_gtDisp[mask], reduction='mean')

        return loss

    def __call__(self, estDisp, gtDisp):
        if not isinstance(estDisp, (list, tuple)):
            estDisp = [estDisp]

        if self.weights is None:
            self.weights = [1.0] * len(estDisp)

        # compute loss for per level
        loss_all_level = []
        for est_disp_per_lvl in estDisp:
            loss_all_level.append(
                self.loss_per_level(est_disp_per_lvl, gtDisp)
            )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "l1_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'SmoothL1Loss'
