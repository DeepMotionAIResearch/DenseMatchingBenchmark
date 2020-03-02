import warnings
from collections import abc as container_abcs

import numpy as np

import torch

from dmb.modeling.stereo.layers.inverse_warp import inverse_warp
from dmb.data.datasets.evaluation.stereo.pixel_error import calc_error


def remove_padding(batch, size):
    """
    Usually, the SceneFlow image size is [540, 960], and we often pad it to [544, 960] evaluation,
    What's more, for KITTI, the image size is pad to [384, 1248]
    Details refer to dmb.data.transforms.stereo_trans.Pad
    Here, we mainly remove the padding from the estimated tensor, such as disparity map
    Args:
        batch (torch.Tensor): in [BatchSize, Channel, Height, Width] layout
        size (list, tuple): the last two dimensions are desired [Height, Width]
    """
    error_msg = "batch must contain tensors, dicts or lists; found {}"
    if isinstance(batch, torch.Tensor):
        # Crop batch to desired size
        # For stereo, we often pad top and right of the image
        pad_top = batch.shape[-2] - size[-2]
        # pad_right = batch.shape[-1] - size[-1]
        if pad_top >= 0:
            batch = batch[:, :, pad_top:, :size[-1]]

        return batch
    elif isinstance(batch, container_abcs.Mapping):
        return {key: remove_padding(batch[key], size) for key in batch}
    elif isinstance(batch, container_abcs.Sequence):
        return [remove_padding(samples, size) for samples in batch]

    raise TypeError((error_msg.format(type(batch))))


def do_evaluation(est_disp, gt_disp, lb, ub):
    """
    Do pixel error evaluation. (See KITTI evaluation protocols for details.)
    Args:
        est_disp, (Tensor): estimated disparity map, in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        gt_disp, (Tensor): ground truth disparity map, in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        lb, (scalar): the lower bound of disparity you want to mask out
        ub, (scalar): the upper bound of disparity you want to mask out

    Returns:
        error_dict (dict): the error of 1px, 2px, 3px, 5px, in percent,
            range [0,100] and average error epe
    """
    error_dict = {}
    if est_disp is None:
        warnings.warn('Estimated disparity map is None')
        return error_dict
    if gt_disp is None:
        warnings.warn('Reference ground truth disparity map is None')
        return error_dict

    if torch.is_tensor(est_disp):
        est_disp = est_disp.clone().cpu()

    if torch.is_tensor(gt_disp):
        gt_disp = gt_disp.clone().cpu()

    error_dict = calc_error(est_disp, gt_disp, lb=lb, ub=ub)

    return error_dict


def do_occlusion_evaluation(est_disp, ref_gt_disp, target_gt_disp, lb, ub):
    """
    Do occlusoin evaluation.
    Args:
        est_disp: estimated disparity map, in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        ref_gt_disp: reference(left) ground truth disparity map, in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        target_gt_disp: target(right) ground truth disparity map, in [BatchSize, Channel, Height, Width] or
            [BatchSize, Height, Width] or [Height, Width] layout
        lb, (scalar): the lower bound of disparity you want to mask out
        ub, (scalar): the upper bound of disparity you want to mask out

    Returns:

    """
    error_dict = {}
    if est_disp is None:
        warnings.warn('Estimated disparity map is None, expected given')
        return error_dict
    if ref_gt_disp is None:
        warnings.warn('Reference ground truth disparity map is None, expected given')
        return error_dict
    if target_gt_disp is None:
        warnings.warn('Target ground truth disparity map is None, expected given')
        return error_dict

    if torch.is_tensor(est_disp):
        est_disp = est_disp.clone().cpu()
    if torch.is_tensor(ref_gt_disp):
        ref_gt_disp = ref_gt_disp.clone().cpu()
    if torch.is_tensor(target_gt_disp):
        target_gt_disp = target_gt_disp.clone().cpu()

    warp_ref_gt_disp = inverse_warp(target_gt_disp.clone(), -ref_gt_disp.clone())
    theta = 1.0
    eps = 1e-6
    occlusion = (
            (torch.abs(warp_ref_gt_disp.clone() - ref_gt_disp.clone()) > theta) |
            (torch.abs(warp_ref_gt_disp.clone()) < eps)
    ).prod(dim=1, keepdim=True).type_as(ref_gt_disp)
    occlusion = occlusion.clamp(0, 1)

    occlusion_error_dict = calc_error(
        est_disp.clone() * occlusion.clone(),
        ref_gt_disp.clone() * occlusion.clone(),
        lb=lb, ub=ub
    )
    for key in occlusion_error_dict.keys():
        error_dict['occ_' + key] = occlusion_error_dict[key]

    not_occlusion = 1.0 - occlusion
    not_occlusion_error_dict = calc_error(
        est_disp.clone() * not_occlusion.clone(),
        ref_gt_disp.clone() * not_occlusion.clone(),
        lb=lb, ub=ub
    )
    for key in not_occlusion_error_dict.keys():
        error_dict['noc_' + key] = not_occlusion_error_dict[key]

    return error_dict
