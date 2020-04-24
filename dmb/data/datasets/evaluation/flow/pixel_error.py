import numpy as np

import torch

def zero_mask(input, eps=1e-12):
    mask = abs(input) < eps
    return mask

def calc_error(est_flow=None, gt_flow=None, sparse=False):
    """
    Args:
        est_flow (Tensor): in [BatchSize, 2, Height, Width] or
                              [2, Height, Width] layout
        gt_flow (Tensor): in [BatchSize, 2, Height, Width] or
                             [2, Height, Width] layout
        sparse, (bool): whether the given flow is sparse, default False
    Output:
        dict: the error of 1px, 2px, 3px, 5px, in percent,
            range [0,100] and average error epe
    """
    error1 = torch.Tensor([0.])
    error2 = torch.Tensor([0.])
    error3 = torch.Tensor([0.])
    error5 = torch.Tensor([0.])
    epe = torch.Tensor([0.])

    if (not torch.is_tensor(est_flow)) or (not torch.is_tensor(gt_flow)):
        return {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }

    assert torch.is_tensor(est_flow) and torch.is_tensor(gt_flow)
    assert est_flow.shape == gt_flow.shape

    est_flow = est_flow.clone().cpu()
    gt_flow = gt_flow.clone().cpu()
    if len(gt_flow.shape) == 3:
        gt_flow = gt_flow.unsqueeze(0)
        est_flow = est_flow.unsqueeze(0)

    assert gt_flow.shape[1] == 2, "flow should have horizontal and vertical dimension, " \
                                  "but got {}".format(gt_flow.shape[1])

    # [B, 1, H, W]
    gt_u, gt_v = gt_flow[:, 0:1, :, :], gt_flow[:, 1:2, :, :]
    est_u, est_v = est_flow[:, 0:1, :, :], est_flow[:, 1:2, :, :]

    # get valid mask
    # [B, 1, H, W]
    mask = torch.ones(gt_u.shape, dtype=torch.bool)
    if sparse:
        mask = mask & (~(zero_mask(gt_u) & zero_mask(gt_v)))
    mask = mask & (~(torch.isnan(gt_u) | torch.isnan(gt_v)))
    mask.detach_()
    if abs(mask.float().sum()) < 1.0:
        return {
            '1px': error1 * 100,
            '2px': error2 * 100,
            '3px': error3 * 100,
            '5px': error5 * 100,
            'epe': epe
        }

    gt_u = gt_u[mask]
    gt_v = gt_v[mask]
    est_u = est_u[mask]
    est_v = est_v[mask]

    abs_error = torch.sqrt((gt_u - est_u)**2 + (gt_v - est_v)**2)
    total_num = mask.float().sum()

    error1 = torch.sum(torch.gt(abs_error, 1).float()) / total_num
    error2 = torch.sum(torch.gt(abs_error, 2).float()) / total_num
    error3 = torch.sum(torch.gt(abs_error, 3).float()) / total_num
    error5 = torch.sum(torch.gt(abs_error, 5).float()) / total_num
    epe = abs_error.float().mean()

    return {
        '1px': error1 * 100,
        '2px': error2 * 100,
        '3px': error3 * 100,
        '5px': error5 * 100,
        'epe': epe
    }
