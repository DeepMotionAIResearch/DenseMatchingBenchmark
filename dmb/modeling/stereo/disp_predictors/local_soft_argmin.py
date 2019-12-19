import torch
import torch.nn.functional as F


def local_soft_argmin(cost_volume, sigma):
    # type: (torch.Tensor, int) -> torch.Tensor
    r"""Implementation of a local soft argmin
    Arguments:
        cost_volume (torch.Tensor): in [BatchSize, MaxDisparity, Height, Width] layout
        sigma (int): select => d':|d'-d|<=sigma, d' = argmax( P(d) for d in 1:maxDisp )
    Returns:
        disp_map (torch.Tensor): a disparity map regressed from cost volume, in [B, 1, W, H] layout
    """
    if cost_volume.dim() != 4:
        raise ValueError('expected 4D input (got {}D input)'
                         .format(cost_volume.dim()))

    if not isinstance(sigma, int):
        raise TypeError('argument \'sigma\' must be int, not {}'.format(type(sigma)))

    # grab max disparity
    max_disp = cost_volume.shape[1]
    N = cost_volume.size()[0]
    H = cost_volume.size()[2]
    W = cost_volume.size()[3]

    # d':|d'-d|<=sigma, d' = argmax( P(d) for d in 1:maxDisp ), (BatchSize, 1, Height, Width)
    index = torch.argmax(cost_volume, dim=1, keepdim=True)
    interval = torch.linspace(-sigma, sigma, 2 * sigma + 1).type_as(index).to(cost_volume.device)
    interval = interval.repeat(N, H, W, 1).permute(0, 3, 1, 2).contiguous()
    # (BatchSize, 2*sigma+1, Height, Width)
    index_group = (index + interval)

    # get mask in [0, max_disp)
    mask = ((index_group >= 0) & (index_group < max_disp)).detach().type_as(cost_volume)
    index_group = index_group.clamp(0, max_disp - 1)

    # gather values in the index_group
    disp_map = torch.gather(cost_volume, dim=1, index=index_group)

    # convert index_group from torch.LongTensor to torch.FloatTensor
    index_group = index_group.type_as(cost_volume)

    # d * P(d), and mask out index out of [0, max_disp), (BatchSize, 1, Height, Width)
    # if index in [0, max_disp), keep the original disparity value, otherwise -10000.0, as e(-10000.0) approximate 0.0
    disp_map = F.softmax((disp_map * mask + (1 - mask) * (-10000.0)), dim=1)
    disp_map = (disp_map * index_group).sum(dim=1, keepdim=True)

    return disp_map
