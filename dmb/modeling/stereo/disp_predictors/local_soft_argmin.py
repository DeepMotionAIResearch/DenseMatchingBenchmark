import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalSoftArgmin(nn.Module):
    """
    An implementation of local soft argmin.
    Attention:
        1. this function is not differential
        2. cost volume will be normalized with softmax, so 'normalize' is invalid
    Args:
        max_disp, (int): under the scale of feature used,
            often equals to (end disp - start disp + 1), the maximum searching range of disparity
        radius (int): select d:|d'-d|<=radius, d' = argmax( P(d) for d in dim[1] )
        start_disp (int): the start searching disparity index, usually be 0
        dilation (int): the step between near disparity index
        radius_dilation (int): the step between near disparity index when local sampling
        alpha (float or int): a factor will times with cost_volume
            details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
        normalize (bool): whether apply softmax on cost_volume, default True

    Inputs:
        cost_volume (Tensor): the matching cost after regularization,
            in [B, disp_sample_number, W, H] layout
        disp_sample (optional, Tensor): the estimated disparity samples,
            in [BatchSize, disp_sample_number, Height, Width] layout. NOT USED!

    Returns:
        disp_map (Tensor): a disparity map regressed from cost volume,
            in [B, 1, W, H] layout
    """

    def __init__(self, max_disp, radius, start_disp=0, dilation=1, radius_dilation=1, alpha=1.0, normalize=True):
        super(LocalSoftArgmin, self).__init__()
        self.max_disp = max_disp
        self.radius = radius
        self.start_disp = start_disp
        self.dilation = dilation
        self.radius_dilation = radius_dilation
        self.end_disp = start_disp + max_disp - 1
        self.disp_sample_number = (max_disp + dilation - 1) // dilation

        self.alpha = alpha
        self.normalize = normalize


    def forward(self, cost_volume, disp_sample=None):

        # note, cost volume direct represent similarity
        # 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.

        # grab index with max similarity

        B = cost_volume.size()[0]

        D = cost_volume.size()[1]
        assert D == self.disp_sample_number, 'Number of disparity sample should be same' \
                                             'with predicted disparity number in cost volume!'

        H = cost_volume.size()[2]
        W = cost_volume.size()[3]

        # d':|d'-d|<=sigma, d' = argmax( C(d) for d in dim[1] ), (BatchSize, 1, Height, Width)
        # it's only the index for array, not real disparity index
        max_index = torch.argmax(cost_volume, dim=1, keepdim=True)

        # sample near the index of max similarity, get [2 * radius + 1]
        # for example, if dilation=2, disp_sample_radius =2, we will get (-4, -2, 0, 2, 4)
        interval = torch.linspace(-self.radius * self.radius_dilation,
                                  self.radius * self.radius_dilation,
                                  2 * self.radius + 1).long().to(cost_volume.device)
        # (BatchSize, 2 * radius + 1, Height, Width)
        interval = interval.repeat(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

        # (BatchSize, 2*radius+1, Height, Width)
        index_group = (max_index + interval)


        # get mask in [0, D-1],
        # (BatchSize, 2*radius+1, Height, Width)
        mask = ((index_group >= 0) & (index_group <= D-1)).detach().type_as(cost_volume)
        index_group = index_group.clamp(0, D-1)

        # gather values in cost_volume which index = index_group,
        # (BatchSize, 2*radius+1, Height, Width)
        gathered_cost_volume = torch.gather(cost_volume, dim=1, index=index_group)

        # convert index_group from torch.LongTensor to torch.FloatTensor
        index_group = index_group.type_as(cost_volume)

        # convert to real disparity sample index
        disp_sample = self.start_disp + index_group * self.dilation

        # d * P(d), and mask out index out of (start_disp, end_disp), (BatchSize, 1, Height, Width)
        # if index in (start_disp, end_disp), keep the original disparity value, otherwise -10000.0, as e(-10000.0) approximate 0.0
        # scale cost volume with alpha
        gathered_cost_volume = gathered_cost_volume * self.alpha

        # (BatchSize, 2 * radius + 1, Height, Width)
        gathered_prob_volume = F.softmax((gathered_cost_volume * mask + (1 - mask) * (-10000.0 * self.alpha)), dim=1)

        # (BatchSize, 1, Height, Width)
        disp_map = (gathered_prob_volume * disp_sample).sum(dim=1, keepdim=True)

        return disp_map

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Max Disparity: {}\n'.format(self.max_disp)
        repr_str += ' ' * 4 + 'Local disparity sample radius: {}\n'.format(self.radius)
        repr_str += ' ' * 4 + 'Start disparity: {}\n'.format(self.start_disp)
        repr_str += ' ' * 4 + 'Dilation rate: {}\n'.format(self.dilation)
        repr_str += ' ' * 4 + 'Local disparity sample dilation rate: {}\n'.format(self.radius_dilation)
        repr_str += ' ' * 4 + 'Alpha: {}\n'.format(self.alpha)
        repr_str += ' ' * 4 + 'Normalize: {}\n'.format(self.normalize)

        return repr_str

    @property
    def name(self):
        return 'LocalSoftArgmin'


