import os
import os.path as osp
import warnings

import numpy as np
from imageio import imread
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

import torch

import mmcv
from mmcv import mkdir_or_exist
from mmcv.runner import load_checkpoint

import dmb.data.transforms as T
from dmb.modeling.stereo import build_stereo_model as build_model
from dmb.data.datasets.utils import load_scene_flow_disp
from dmb.visualization.stereo import ShowResultTool

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_pfm_file(filename):
    return filename.endswith('.pfm')


def load_disp(item, filename):
    if filename in item.keys() and item[filename] is not None:
        if is_image_file(item[filename]):
            Disp = imread(item[filename]).astype(np.float32)
        elif is_pfm_file(item[filename]):
            Disp = load_scene_flow_disp(item[filename])
    else:
        Disp = None

    return Disp


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
                     scale_factor=None,
                     disp_scale=1.0):
    """Inference image(s) with the stereo model.
    Args:
        model (nn.Module): The loaded detector.
        batchesDict:
    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_transform = []
    img_transform.append(T.ToTensor())
    if pad_to_shape is not None:
        img_transform.append(T.StereoPad(pad_to_shape))
    if crop_shape is not None:
        img_transform.append(T.CenterCrop(crop_shape))
    img_transform.append(T.Normalize(mean, std))
    img_transform = T.Compose(img_transform)

    model.cfg.log_dir = log_dir
    model.cfg.pad_to_shape = pad_to_shape
    model.cfg.crop_shape = crop_shape
    model.scale_factor = scale_factor
    model.cfg.disp_scale = disp_scale

    device = next(model.parameters()).device  # model device
    for batchDict in batchesDict:
        _inference_single(model, batchDict, img_transform, device)


def _prepare_data(item, img_transform, cfg, device):
    origLeftImage = imread(item['left_image_path']).astype(np.float32)[:3]
    origRightImage = imread(item['right_image_path']).astype(np.float32)[:3]
    origLeftDisp = load_disp(item, 'left_disp_map_path') / cfg.disp_scale
    origRightDisp = load_disp(item, 'right_disp_map_path') / cfg.disp_scale

    origSample = {'leftImage': origLeftImage,
                  'rightImage': origRightImage,
                  'leftDisp': origLeftDisp,
                  'rightDisp': origRightDisp}

    if cfg.crop_shape is not None:
        origSample = T.CenterCrop(cfg.crop_shape)(origSample)

    leftImage = origSample['leftImage'].copy().transpose(2, 0, 1)
    rightImage = origSample['rightImage'].copy().transpose(2, 0, 1)
    leftDisp = origSample['leftDisp'].copy()[np.newaxis, ...]
    rightDisp = origSample['rightDisp'].copy()[np.newaxis, ...]

    processSample = {'leftImage': leftImage,
                     'rightImage': rightImage,
                     'leftDisp': leftDisp,
                     'rightDisp': rightDisp}

    h, w = leftImage.shape[1], leftImage.shape[2]
    original_size = (h, w)

    return {
        'leftImage': leftImage,
        'rightImage': rightImage,
        'leftDisp': leftDisp,
        'rightDisp': rightDisp,
        'original_size': original_size,
    }

    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, batchDict, img_transform, device):
    data = _prepare_data(batchDict, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(data)
    return result


def _inference_generator(model, imgPairs, img_transform, device):
    for imgPair in imgPairs:
        yield _inference_single(model, imgPair, img_transform, device)


def save_result(result, out_dir, image_name):
    result_tool = ShowResultTool()
    result = result_tool(result, color_map='gray', bins=100)

    if 'GrayDisparity' in result.keys():
        grayEstDisp = result['GrayDisparity']
        gray_save_path = osp.join(out_dir, 'disp_0')
        mkdir_or_exist(gray_save_path)
        skimage.io.imsave(osp.join(gray_save_path, image_name), (grayEstDisp * 256).astype('uint16'))

    if 'ColorDisparity' in result.keys():
        colorEstDisp = result['ColorDisparity']
        color_save_path = osp.join(out_dir, 'color_disp')
        mkdir_or_exist(color_save_path)
        plt.imsave(osp.join(color_save_path, image_name), colorEstDisp, cmap=plt.cm.hot)

    if 'GroupColor' in result.keys():
        group_save_path = os.path.join(out_dir, 'group_disp')
        mkdir_or_exist(group_save_path)
        plt.imsave(osp.join(group_save_path, image_name), result['GroupColor'], cmap=plt.cm.hot)

    if 'ColorConfidence' in result.keys():
        conf_save_path = os.path.join(out_dir, 'confidence')
        mkdir_or_exist(conf_save_path)
        plt.imsave(osp.join(conf_save_path, image_name), result['ColorConfidence'])
