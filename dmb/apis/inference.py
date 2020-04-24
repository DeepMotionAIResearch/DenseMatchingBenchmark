import os
import os.path as osp
import warnings
from collections import abc as container_abcs

import numpy as np
from imageio import imread

import torch
import torch.nn.functional as F

import mmcv
from mmcv import mkdir_or_exist
from mmcv.runner import load_checkpoint

from dmb.data.transforms import stereo_trans as T
from dmb.data.transforms.transforms import Compose
from dmb.modeling import build_model
from dmb.data.datasets.utils import load_scene_flow_disp
from dmb.data.datasets.evaluation.stereo.eval import remove_padding

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_pfm_file(filename):
    return filename.endswith('.pfm')


def load_disp(item, filename, disp_div_factor=1.0):
    Disp = None
    if filename in item.keys() and item[filename] is not None:
        if is_image_file(item[filename]):
            Disp = imread(item[filename]).astype(np.float32) / disp_div_factor
        elif is_pfm_file(item[filename]):
            Disp = load_scene_flow_disp(item[filename]).astype(np.float32) / disp_div_factor
        else:
            raise NotImplementedError

    return Disp


def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]

    raise TypeError((error_msg.format(type(tensor))))


def init_model(config, checkpoint=None, device='cuda:0'):
    """
    Initialize a stereo model from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
    Returns:
        nn.Module: The constructed stereo model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))

    model = build_model(config)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_stereo(model,
                     batchesDict,
                     log_dir,
                     pad_to_shape=None,
                     crop_shape=None,
                     scale_factor=1.0,
                     disp_div_factor=1.0,
                     device='cuda:0'):
    """Inference image(s) with the stereo model.
    Args:
        model (nn.Module): The loaded model.
        batchesDict (list of dict): a dict must contain: left_image_path, right_image_path;
                                   optional contain: left_disp_map_path, right_disp_map_path
        log_dir (str): result saving root directory
        pad_to_shape (tuple): the shape of image after pad -- (H, W)
        crop_shape (tuple): the shape of image after crop -- (H, W)
        scale_factor (int, float): the down sample or up sample scale of images
        disp_div_factor (int, float): if disparity map given, after reading the disparity map,
                    often have to divide a scale to get the real disparity value, e.g. 256 in KITTI
    Notes:
        Given left and right image path,
        1st: read images
        2nd: pad or crop images to a given shape
        3rd: down sample or up sample the images to the given shape
        4th: model inference
        5th: inversely down sample or up sample result
        finally: if pad, retrieve to original shape; otherwise, nothing will be done
    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    img_transform = []
    img_transform.append(T.ToTensor())
    if pad_to_shape is not None:
        assert crop_shape is None
        img_transform.append(T.StereoPad(pad_to_shape))
    if crop_shape is not None:
        assert pad_to_shape is None
        img_transform.append(T.CenterCrop(crop_shape))
    img_transform.append(T.Normalize(mean, std))
    img_transform = Compose(img_transform)

    model.cfg.update(
        {
            'log_dir': log_dir,
            'pad_to_shape': pad_to_shape,
            'crop_shape': crop_shape,
            'scale_factor': scale_factor,
            'disp_div_factor': disp_div_factor
        }
    )

    device = next(model.parameters()).device  # model device
    for batchDict in batchesDict:
        _inference_single(model, batchDict, img_transform, device)


def _prepare_data(item, img_transform, cfg, device):
    oriLeftImage = imread(item['left_image_path'])[:, :, :3].astype(np.float32)
    oriRightImage = imread(item['right_image_path'])[:, :, :3].astype(np.float32)
    oriLeftDisp = load_disp(item, 'left_disp_map_path', cfg.disp_div_factor)
    oriRightDisp = load_disp(item, 'right_disp_map_path', cfg.disp_div_factor)

    oriSample = {'leftImage': oriLeftImage,
                 'rightImage': oriRightImage,
                 'leftDisp': oriLeftDisp,
                 'rightDisp': oriRightDisp}

    leftImage = oriLeftImage.copy().transpose(2, 0, 1)
    rightImage = oriRightImage.copy().transpose(2, 0, 1)
    leftDisp = None
    rightDisp = None
    if oriLeftDisp is not None:
        leftDisp = oriLeftDisp.copy()[np.newaxis, ...]
    if oriRightDisp is not None:
        rightDisp = oriRightDisp.copy()[np.newaxis, ...]

    h, w = leftImage.shape[1], leftImage.shape[2]
    original_size = (h, w)

    procSample = {'leftImage': leftImage,
                  'rightImage': rightImage,
                  'leftDisp': leftDisp,
                  'rightDisp': rightDisp,
                  'original_size': original_size,
                  }

    procSample = img_transform(procSample)

    scale_factor = cfg.scale_factor
    for k, v in procSample.items():
        if torch.is_tensor(v):
            v = v.unsqueeze(0)
            if 'Disp' in k:
                v = F.interpolate(v*scale_factor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            else:
                v = F.interpolate(v, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            procSample[k] = v.to(device)

    return procSample, oriSample


def _inference_single(model, batchDict, img_transform, device):
    cfg = model.cfg.copy()
    procData, oriData = _prepare_data(batchDict, img_transform, cfg, device)
    with torch.no_grad():
        result, _ = model(procData)
    result = to_cpu(result)

    assert isinstance(result, dict)

    for k, v in result.items():
        assert isinstance(v, (tuple, list))
        for i in range(len(v)):
            vv = v[i]
            if torch.is_tensor(vv):
                # inverse up/down sample
                vv = F.interpolate(vv*1.0/cfg.scale_factor, scale_factor=1.0/cfg.scale_factor, mode='bilinear', align_corners=False)
                ori_size = procData['original_size']
                if cfg.pad_to_shape is not None:
                    vv = remove_padding(vv, ori_size)
                v[i] = vv
        result[k] = v

    logData = {
        'Result': result,
        'OriginalData': oriData,
    }

    save_root = osp.join(cfg.log_dir, batchDict['left_image_path'].split('/')[-1].split('.')[0])
    mkdir_or_exist(save_root)
    save_path = osp.join(save_root, 'result.pkl')
    print('Result of {} will be saved to {}!'.format(batchDict['left_image_path'].split('/')[-1], save_path))

    mmcv.dump(logData, save_path)

    return logData

