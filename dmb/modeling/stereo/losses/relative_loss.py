import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeLoss(object):
    """
    This is an implementation the relative depth loss proposed in paper "Surface Normals in the Wild"
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
        lable (Tensor or list of Tensor): the ground truth relative rank label,
            details can be referred in paper. in [BatchSize, 1, Height, Width] layout.
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

    def loss_per_level(self, estDisp, gtDisp, label):
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
            print('Relative loss: there is no point\'s disparity is in ({},{})!'.format(self.start_disp,
                                                                                        self.max_disp / scale))
            loss = (torch.abs(estDisp - scaled_gtDisp) * mask.float()).mean()
            return loss

        # relative loss
        valid_pixel_number = mask.float().sum()
        diff = scaled_gtDisp[mask] - estDisp[mask]
        label = label[mask]
        # some value which is over large for torch.exp() is not suitable for soft margin loss
        # get absolute value great than 66
        over_large_mask = torch.gt(torch.abs(diff), 66)
        over_large_diff = diff[over_large_mask]
        # get absolute value smaller than 66
        proper_mask = torch.le(torch.abs(diff), 66)
        proper_diff = diff[proper_mask]
        # generate lable for soft margin loss
        label = label[proper_mask]
        loss = F.soft_margin_loss(proper_diff, label, reduction='sum') + torch.abs(over_large_diff).sum()
        loss = loss / valid_pixel_number

        return loss

    def __call__(self, estDisp, gtDisp, label):
        if not isinstance(estDisp, (list, tuple)):
            estDisp = [estDisp]

        if self.weights is None:
            self.weights = [1.0] * len(estDisp)

        if not isinstance(label, (list, tuple)):
            label = [label] * len(estDisp)

        # compute loss for per level
        loss_all_level = []
        for est_disp_per_lvl, label_per_lvl in zip(estDisp, label):
            loss_all_level.append(
                self.loss_per_level(est_disp_per_lvl, gtDisp, label_per_lvl)
            )

        # re-weight loss per level
        weighted_loss_all_level = dict()
        for i, loss_per_level in enumerate(loss_all_level):
            name = "relative_loss_lvl{}".format(i)
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
        return 'RelativeLoss'
