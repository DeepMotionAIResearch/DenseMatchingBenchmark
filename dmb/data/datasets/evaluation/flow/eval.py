import warnings
from collections import abc as container_abcs

import torch

from dmb.data.datasets.evaluation.flow.pixel_error import calc_error


def remove_padding(batch, size):
    """
    Usually, the SceneFlow image size is [540, 960], and we often pad it to [544, 960] evaluation,
    What's more, for KITTI, the image size is pad to [384, 1248]
    Here, we mainly remove the padding from the estimated tensor, such as flow map
    Args:
        batch (torch.Tensor): in [BatchSize, Channel, Height, Width] layout
        size (list, tuple): the last two dimensions are desired [Height, Width]
    """
    error_msg = "batch must contain tensors, dicts or lists; found {}"
    if isinstance(batch, torch.Tensor):
        # Crop batch to desired size
        # For flow, we often pad image around and keep it in the center
        assert batch.shape[-2] >= size[-2] and batch.shape[-1] >= size[-1]

        pad_top = (batch.shape[-2] - size[-2])//2
        pad_left = (batch.shape[-1] - size[-1])//2
        # pad_right = batch.shape[-1] - size[-1]
        batch = batch[:, :, pad_top:, pad_left:]

        return batch
    elif isinstance(batch, container_abcs.Mapping):
        return {key: remove_padding(batch[key], size) for key in batch}
    elif isinstance(batch, container_abcs.Sequence):
        return [remove_padding(samples, size) for samples in batch]

    raise TypeError((error_msg.format(type(batch))))


def do_evaluation(est_flow, gt_flow, sparse=False):
    """
    Do pixel error evaluation. (See KITTI evaluation protocols for details.)
    Args:
        est_flow, (Tensor): estimated flow map, in [BatchSize, 2, Height, Width] or
            [2, Height, Width] layout
        gt_flow, (Tensor): ground truth flow map, in [BatchSize, 2, Height, Width] or
            [2, Height, Width]layout
        sparse, (bool): whether the given flow is sparse, default False

    Returns:
        error_dict (dict): the error of 1px, 2px, 3px, 5px, in percent,
            range [0,100] and average error epe
    """
    error_dict = {}
    if est_flow is None:
        warnings.warn('Estimated flow map is None')
        return error_dict
    if gt_flow is None:
        warnings.warn('Reference ground truth flow map is None')
        return error_dict

    if torch.is_tensor(est_flow):
        est_flow = est_flow.clone().cpu()

    if torch.is_tensor(gt_flow):
        gt_flow = gt_flow.clone().cpu()

    error_dict = calc_error(est_flow, gt_flow, sparse=sparse)

    return error_dict


