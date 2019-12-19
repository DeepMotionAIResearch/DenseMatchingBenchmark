import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
from mmcv import mkdir_or_exist

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def getKITTI2015Metas(root, Type, mode, is_full=True):
    r"""
    Arguments:
        root, (str): the dataset root
        Type, (str): data type: training, testing
        mode, (str): phase: training, testing or evaluating
        is_full, (bool): whether to set the whole dataset as training dataset, or split into training/evaluating
    """
    Metas = []

    imageNames = [img for img in os.listdir(osp.join(root, Type, 'image_2')) if img.find('_10') > -1]
    imageNames.sort()

    for imageName in imageNames:
        meta = dict(
            left_image_path=osp.join(
                Type, 'image_2', imageName
            ),
            right_image_path=osp.join(
                Type, 'image_3', imageName
            ),
            left_disp_map_path=osp.join(
                Type, 'disp_occ_0', imageName
            ) if Type == 'training' else None,  # testing dataset has no ground truth left disparity map
            right_disp_map_path=osp.join(
                Type, 'disp_occ_1', imageName
            ) if Type == 'training' else None,  # testing dataset has no ground truth left disparity map
        )
        Metas.append(meta)

    # if not is_full, split the 200 images into training and evaluating
    eval_list = [1, 3, 6, 20, 26, 35, 38, 41, 43, 44,
                 49, 60, 67, 70, 81, 84, 89, 97, 109, 119,
                 122, 123, 129, 130, 132, 134, 141, 144, 152, 158,
                 159, 165, 171, 174, 179, 182, 184, 186, 187, 196]
    if not is_full and Type == 'training':
        subMetas = []
        for idx in range(200):
            if mode == 'evaluating' and idx in eval_list:
                subMetas.append(Metas[idx])
            if mode == 'training' and idx not in eval_list:
                subMetas.append(Metas[idx])
        Metas = subMetas

    return Metas


def check(root, Metas):
    for meta in tqdm(Metas):
        for k, v in meta.items():
            if v is not None:
                assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
                assert is_image_file(v), 'trainMetas:{} is not a image file'.format(v)


def build_annoFile(root, save_annotation_root, is_full=True):
    """
    Build annotation files for Scene Flow Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    mkdir_or_exist(save_annotation_root)

    trainMetas = getKITTI2015Metas(root, 'training', mode='training', is_full=is_full)
    evalMetas = getKITTI2015Metas(root, 'training', mode='evaluating', is_full=is_full)
    testMetas = getKITTI2015Metas(root, 'testing', mode='testing', is_full=True)

    check(root, trainMetas)
    check(root, evalMetas)
    check(root, testMetas)

    info_str = 'KITTI-2015 Dataset contains:\n' \
               '    {:5d}   training samples \n' \
               '    {:5d}   validation samples \n' \
               '    {:5d}   testing samples'.format(len(trainMetas), len(evalMetas), len(testMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, name + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    prefix = 'full_' if is_full else 'split_'
    make_json(name=prefix + 'train', metas=trainMetas)
    make_json(name=prefix + 'eval', metas=evalMetas)
    make_json(name=prefix + 'test', metas=testMetas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KITTI 2015 Data PreProcess.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="root of data",
        type=str,
    )
    parser.add_argument(
        "--save-annotation-root",
        default='./',
        help="save root of generated annotation file",
        type=str,
    )
    parser.add_argument(
        "--is-full",
        action='store_true',
        help="whether set all images to be training or evaluating dataset",
    )
    args = parser.parse_args()

    build_annoFile(args.data_root, args.save_annotation_root, args.is_full)
