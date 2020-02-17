import torch
import torch.nn as nn
import torch.nn.functional as F

def quantile_loss(minEstDisp, maxEstDisp, gtDisp, max_disp, start_disp=0, weight=1.0, theta=0.05):
    """
    An implementation of quantile loss proposed in DeepPruner
    Details refer to: https://github.com/uber-research/DeepPruner/blob/master/deeppruner/loss_evaluation.py

    Inputs:
        minEstDisp, (Tensor): the estimated min disparity map, i.e. the lower bound of disparity samples,
                              in [BatchSize, 1, Height, Width] layout.
        maxEstDisp, (Tensor): the estimated max disparity map, i.e. the upper bound of disparity samples
                              in [BatchSize, 1, Height, Width] layout.
        gtDisp, (Tensor): the ground truth disparity map,
                          in [BatchSize, 1, Height, Width] layout.
        max_disp (int): the max of Disparity. default is 192
        start_disp (int): the start searching disparity index, usually be 0
        weight (int, float): the weight of quantile loss
        theta (float): the balancing scalar, 0 < theta < 0.05


    """
    # get valid ground truth disparity
    mask = (gtDisp > start_disp) & (gtDisp < (start_disp + max_disp))

    # forces min_disparity to be equal or slightly lower than the ground truth disparity
    min_mask = ((gtDisp[mask] -minEstDisp[mask]) < 0).float()
    # if x < 0, x * (-0.95); if x > 0, x * 0.05
    min_loss = (gtDisp[mask] - minEstDisp[mask]) * (theta - min_mask)
    min_loss = min_loss.mean()

    # forces max_disparity to be equal or slightly larger than the ground truth disparity
    max_mask = ((gtDisp[mask] - maxEstDisp[mask]) < 0).float()
    # if x < 0, x * (-0.05); if x > 0, x * 0.95
    max_loss = (gtDisp[mask] - maxEstDisp[mask]) * ((1 - theta) - max_mask)
    max_loss = max_loss.mean()

    total_loss = (min_loss + max_loss) * weight

    return total_loss