import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: remove nn.Conv3d, use F.conv3d directly.
class FasterSoftArgmin(nn.Module):
    """
    A faster implementation of soft argmin.
    details can refer to dmb.modeling.stereo.disp_prediction.soft_argmin
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the max searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (int): the step between near disparity index
        normalize (bool): whether apply softmax on cost_volume, default True
        alpha (float or int): a factor will times with cost_volume
            details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
    Inputs:
        cost_volume (Tensor): the matching cost after regularization,
            in [B, disp_sample_number, W, H] layout
    Returns:
        disp_map (Tensor): a disparity map regressed from cost volume,
            in [B, 1, W, H] layout
    """

    def __init__(self, max_disp, alpha=1.0, start_disp=0, dilation=1):
        super(FasterSoftArgmin, self).__init__()
        self.max_disp = max_disp
        self.alpha = alpha

        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + max_disp - 1
        self.disp_sample_number = (max_disp + dilation - 1) // dilation

        self.disp_regression = nn.Conv3d(1, 1, (self.disp_sample_number, 1, 1), 1, 0, bias=False)

        # compute disparity index: (1 ,1, disp_sample_number, 1, 1)
        disp_index = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        ).repeat(1, 1, 1, 1, 1).permute(0, 1, 4, 2, 3).contiguous()
        self.disp_regression.weight.data = disp_index
        self.disp_regression.weight.requires_grad = False

    def forward(self, cost_volume, normalize=True):
        # scale cost volume with alpha
        cost_volume = cost_volume * self.alpha

        if normalize:
            prob_volume = F.softmax(cost_volume, dim=1)
        else:
            prob_volume = cost_volume

        # [B, disp_sample_number, W, H] -> [B, 1, disp_sample_number, W, H]
        prob_volume = prob_volume.unsqueeze(1)

        disp_map = self.disp_regression(prob_volume)
        # [B, 1, 1, W, H] -> [B, 1, W, H]
        disp_map = disp_map.squeeze(1)

        return disp_map
