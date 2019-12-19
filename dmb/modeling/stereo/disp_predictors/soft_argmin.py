import torch
import torch.nn.functional as F


def soft_argmin(cost_volume, normalize=True, temperature=1.0):
    # type: (torch.Tensor, bool, [float, int]) -> torch.Tensor
    r"""Implementation of soft argmin proposed by GC-Net.
    Arguments:
        cost_volume (torch.Tensor): the matching cost after regularization, in [B, max_disp, W, H] layout
        normalize (bool): whether apply softmax on cost_volume, default True
        temperature (float, int): a temperature factor will times with cost_volume
                    details can refer to: https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/
    Returns:
        disp_map (torch.Tensor): a disparity map regressed from cost volume, in [B, 1, W, H] layout
    """
    if cost_volume.dim() != 4:
        raise ValueError(
            'expected 4D input (got {}D input)'.format(cost_volume.dim())
        )
    # grab max disparity
    N, max_disp, H, W = cost_volume.shape

    # generate disparity indexes
    disp_index = torch.linspace(0, max_disp - 1, max_disp).to(cost_volume.device)
    disp_index = disp_index.repeat(N, H, W, 1).permute(0, 3, 1, 2).contiguous()

    # compute probability volume
    # prob_volume: (BatchSize, MaxDisparity, Height, Width)
    cost_volume = cost_volume * temperature
    if normalize:
        prob_volume = F.softmax(cost_volume, dim=1)
    else:
        prob_volume = cost_volume

    # compute disparity: (BatchSize, 1, Height, Width)
    disp_map = torch.sum(prob_volume * disp_index, dim=1, keepdim=True)

    return disp_map
