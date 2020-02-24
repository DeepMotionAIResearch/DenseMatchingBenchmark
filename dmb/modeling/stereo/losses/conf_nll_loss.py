import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceNllLoss(object):
    """
    Args:
        weights (list of float or None): weight for each scale of estCost.
        start_disp (int): the start searching disparity index, usually be 0
        max_disp (int): the max of Disparity. default: 192
        sparse (bool): whether the ground-truth disparity is sparse,
            for example, KITTI is sparse, but SceneFlow is not. default is False
    Inputs:
        estConf (Tensor or list of Tensor): the estimated confidence map,
            in [BatchSize, 1, Height, Width] layout.
        gtDisp (Tensor): the ground truth disparity map,
            in [BatchSize, 1, Height, Width] layout.
    Outputs:
        weighted_loss_all_level (dict of Tensors): the weighted loss of all levels
    """

    def __init__(self, max_disp, start_disp=0, weights=None, sparse=False):
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.weights = weights
        self.sparse = sparse
        if sparse:
            # sparse disparity ==> max_pooling
            self.scale_func = F.adaptive_max_pool2d
        else:
            # dense disparity ==> avg_pooling
            self.scale_func = F.adaptive_avg_pool2d

    def loss_per_level(self, estConf, gtDisp):
        N, C, H, W = estConf.shape
        scaled_gtDisp = gtDisp
        scale = 1.0
        if gtDisp.shape[-2] != H or gtDisp.shape[-1] != W:
            # compute scale per level and scale gtDisp
            scale = gtDisp.shape[-1] / (W * 1.0)
            scaled_gtDisp = gtDisp / scale
            scaled_gtDisp = self.scale_func(scaled_gtDisp, (H, W))

        # mask for valid disparity
        # gt zero and lt max disparity
        mask = (scaled_gtDisp > self.start_disp) & (scaled_gtDisp < (self.max_disp / scale))
        mask = mask.detach_().type_as(gtDisp)

        # NLL loss
        valid_pixel_number = mask.float().sum()
        if valid_pixel_number < 1.0:
            valid_pixel_number = 1.0
        loss = (-1.0 * F.logsigmoid(estConf) * mask).sum() / valid_pixel_number

        return loss

    def __call__(self, estConf, gtDisp):
        if not isinstance(estConf, (list, tuple)):
            estConf = [estConf]

        if self.weights is None:
            self.weights = [1.0] * len(estConf)

        # compute loss for per level
        loss_all_level = [
            self.loss_per_level(est_conf_per_lvl, gtDisp)
            for est_conf_per_lvl in estConf
        ]

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "conf_loss_lvl{}".format(i)
            weighted_loss_all_level[name] = self.weights[i] * loss_per_level

        return weighted_loss_all_level

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Loss weight: {}\n'.format(self.weights)
        repr_str += ' ' * 4 + 'Disparity is sparse: {}\n'.format(self.sparse)

        return repr_str

    @property
    def name(self):
        return 'ConfidenceNLLLoss'
