import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmin(nn.Module):
    """
    An implementation of soft argmin.
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        start_disp (int): the start searching disparity index, usually be 0
        dilation (optional, int): the step between near disparity index
        alpha (float or int): a factor will times with cost_volume
            details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        normalize (bool): whether apply softmax on cost_volume, default True

    Inputs:
        cost_volume (Tensor): the matching cost after regularization,
            in [BatchSize, disp_sample_number, Height, Width] layout
        disp_sample (optional, Tensor): the estimated disparity samples,
            in [BatchSize, disp_sample_number, Height, Width] layout

    Returns:
        disp_map (Tensor): a disparity map regressed from cost volume,
            in [BatchSize, 1, Height, Width] layout
    """

    def __init__(self, max_disp=192, start_disp=0, dilation=1, alpha=1.0, normalize=True):
        super(SoftArgmin, self).__init__()
        self.max_disp = max_disp
        self.start_disp = start_disp
        self.dilation = dilation
        self.end_disp = start_disp + max_disp - 1
        self.disp_sample_number = (max_disp + dilation - 1) // dilation

        self.alpha = alpha
        self.normalize = normalize

        # generate disparity sample, in [disp_sample_number,] layout
        self.disp_sample = torch.linspace(
            self.start_disp, self.end_disp, self.disp_sample_number
        )

    def forward(self, cost_volume, disp_sample=None):

        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        if cost_volume.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(cost_volume.dim()))

        # scale cost volume with alpha
        cost_volume = cost_volume * self.alpha

        if self.normalize:
            prob_volume = F.softmax(cost_volume, dim=1)
        else:
            prob_volume = cost_volume

        B, D, H, W = cost_volume.shape

        if disp_sample is None:
            assert D == self.disp_sample_number, 'The number of disparity samples should be' \
                                                 ' consistent!'
            disp_sample = self.disp_sample.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
            disp_sample = disp_sample.to(cost_volume.device)

        else:
            assert D == disp_sample.shape[1], 'The number of disparity samples should be' \
                                                 ' consistent!'
        # compute disparity: (BatchSize, 1, Height, Width)
        disp_map = torch.sum(prob_volume * disp_sample, dim=1, keepdim=True)

        return disp_map

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Alpha: {}\n'.format(self.alpha)
        repr_str += ' ' * 4 + 'Normalize: {}\n'.format(self.normalize)

        return repr_str

    @property
    def name(self):
        return 'SoftArgmin'
