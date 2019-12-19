import warnings

import numpy as np

import torch


def mask_to_neg(x, mask):
    # if mask=1, keep x, if mask=0, convert x to -1
    x = x * mask + (mask - 1)
    return x


def norm(x):
    x = x / (x.max() - x.min())
    # scale x to [0.05, 0.9] for counting convenient, it doesn't influence the final result
    x = x * 0.9 + 0.05
    return x


def sparsification_plot(est_disp=None, gt_disp=None, est_conf=None, bins=10, lb=None, ub=None):
    """
    Refer to paper: Uncertainty estimates and multi-hypotheses networks for optical flow
    Args:
        est_disp (Tensor): in (..., Height, Width) layout
        gt_disp (Tensor): in (..., Height, Width) layout
        est_conf (Tensor): in (..., Height, Width) layout, we will normalize it to [0,1] for convenient
        bins (int): divide the all pixel into $bins factions, ie each fraction is (100/bins)%
        lb (scaler): the lower bound of disparity you want to mask out
        ub (scaler): the upper bound of disparity you want to mask out
    Output:
        dict: the average error epe when pixels with the lowest confidence are removed gradually
              ideally, the error should monotonically decrease
    """
    assert isinstance(bins, int) and (100 % bins == 0), \
        "bins must be divided by 100, and should be int, but get {} is type {}".format(bins, type(bins))
    error_dict = {}
    percentages = []

    part = 100 // bins
    for i in range(bins + 1):
        percentages.append(part * i)
        error_dict['est_{}'.format(part * i)] = torch.Tensor([0.])
        error_dict['oracle_{}'.format(part * i)] = torch.Tensor([0.])
        error_dict['random_{}'.format(part * i)] = torch.Tensor([0.])

    err_msg = '{} is supposed to be torch.Tensor; find {}'
    if not isinstance(est_disp, torch.Tensor):
        warnings.warn(err_msg.format('Estimated disparity map', type(est_disp)))
    if not isinstance(gt_disp, torch.Tensor):
        warnings.warn(err_msg.format('Ground truth disparity map', type(gt_disp)))
    if not isinstance(est_conf, torch.Tensor):
        warnings.warn(err_msg.format('Estimated confidence map', type(est_conf)))
    if any([not isinstance(est_disp, torch.Tensor), not isinstance(gt_disp, torch.Tensor),
            not isinstance(est_conf, torch.Tensor)]):
        warnings.warn('Input maps contains None, expected given torch.Tensor')
        return error_dict
    if not est_disp.shape == gt_disp.shape:
        warnings.warn('Estimated and ground truth disparity map should have same shape')
    if not est_disp.shape == est_conf.shape:
        warnings.warn('Estimated disparity and confidence map should have same shape')

    if any([not (est_disp.shape == gt_disp.shape), not (est_disp.shape == est_conf.shape)]):
        return error_dict

    est_disp = est_disp.clone().cpu()
    gt_disp = gt_disp.clone().cpu()
    est_conf = est_conf.clone().cpu()

    mask = torch.ones(gt_disp.shape, dtype=torch.uint8)
    if lb is not None:
        mask = mask & (gt_disp > lb)
    if ub is not None:
        mask = mask & (gt_disp < ub)

    mask.detach_()

    total_valid_num = mask.sum()
    if total_valid_num < bins:
        return error_dict

    mask = mask.float()
    est_disp = est_disp * mask
    gt_disp = gt_disp * mask

    abs_error = torch.abs(gt_disp - est_disp)

    # normalize confidence map and error map
    est_conf = norm(est_conf)
    # error is lower the better, but confidence is bigger the better
    neg_norm_abs_error = 1.0 - norm(abs_error)

    # random remove map
    randRemove = torch.rand_like(est_conf)
    randRemove = norm(randRemove)

    # let invalid pixels to -1
    neg_norm_abs_error = mask_to_neg(neg_norm_abs_error, mask)
    est_conf = mask_to_neg(est_conf, mask)
    randRemove = mask_to_neg(randRemove, mask)

    # flatten
    flat_neg_norm_abs_error, _ = neg_norm_abs_error.view(-1).sort()
    flat_est_conf, _ = est_conf.view(-1).sort()
    flat_randRemove, _ = randRemove.view(-1).sort()

    assert (flat_neg_norm_abs_error <= 0).sum() == (flat_est_conf <= 0).sum(), \
        'The number of invalid confidence and disparity should be the same'
    assert (flat_neg_norm_abs_error <= 0).sum() == (flat_randRemove <= 0).sum(), \
        'The number of invalid random map and disparity should be the same'

    start_pointer = (flat_neg_norm_abs_error <= 0).sum()
    part = (total_valid_num - start_pointer - 1) // bins
    pointer_edges = [start_pointer + part * i for i in range(bins + 1)]
    conf_edges = []
    error_edges = []
    rand_edges = []
    for pointer in pointer_edges:
        conf_edges.append(flat_est_conf[pointer])
        error_edges.append(flat_neg_norm_abs_error[pointer])
        rand_edges.append(flat_randRemove[pointer])

    for i in range(bins):
        # kick out the lowest percentages[i]% confidence pixels, and evaluate the left
        conf_mask = (est_conf >= conf_edges[i]).float()
        # kick out the biggest percentages[i]% error pixels, and evaluate the left
        # absolute error is lower is better, it's different from confidence value
        error_mask = (neg_norm_abs_error >= error_edges[i]).float()
        # kick out percentages[i]% random generated value
        rand_mask = (randRemove >= rand_edges[i]).float()

        error_dict['est_{}'.format(percentages[i])] = (abs_error * conf_mask).sum() / (conf_mask.sum())
        error_dict['oracle_{}'.format(percentages[i])] = (abs_error * error_mask).sum() / (error_mask.sum())
        error_dict['random_{}'.format(percentages[i])] = (abs_error * rand_mask).sum() / (rand_mask.sum())

    return error_dict
