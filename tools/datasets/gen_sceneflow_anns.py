import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
from mmcv import mkdir_or_exist


def getFlying3dMetas(root, Type, data_type='clean'):
    Metas = []

    imgDir = 'flyingthings3d/frames_' + data_type + 'pass'
    dispDir = 'flyingthings3d/disparity'
    Parts = ['A', 'B', 'C']

    for Part in Parts:
        partDir = osp.join(root, dispDir, Type, Part)
        idxDirs = os.listdir(partDir)
        for idxDir in idxDirs:
            dispNames = os.listdir(osp.join(partDir, idxDir, 'left'))
            imgNames = ["{}.png".format(name.split('.')[0]) for name in dispNames]
            for imgName, dispName in zip(imgNames, dispNames):
                meta = dict(
                    left_image_path=osp.join(
                        imgDir, Type, Part, idxDir, 'left', imgName
                    ),
                    right_image_path=osp.join(
                        imgDir, Type, Part, idxDir, 'right', imgName
                    ),
                    left_disp_map_path=osp.join(
                        dispDir, Type, Part, idxDir, 'left', dispName
                    ),
                    right_disp_map_path=osp.join(
                        dispDir, Type, Part, idxDir, 'right', dispName
                    ),
                )
                Metas.append(meta)
    return Metas


def getMonkaaMetas(root, data_type='clean'):
    Metas = []

    imgDir = 'Monkaa/frames_' + data_type + 'pass'
    dispDir = 'Monkaa/disparity'

    sceneDirs = os.listdir(osp.join(root, dispDir))

    for sceneDir in sceneDirs:
        dispNames = os.listdir(osp.join(root, dispDir, sceneDir, 'left'))
        imgNames = ["{}.png".format(name.split('.')[0]) for name in dispNames]
        for imgName, dispName in zip(imgNames, dispNames):
            meta = dict(
                left_image_path=osp.join(
                    imgDir, sceneDir, 'left', imgName
                ),
                right_image_path=osp.join(
                    imgDir, sceneDir, 'right', imgName
                ),
                left_disp_map_path=osp.join(
                    dispDir, sceneDir, 'left', dispName
                ),
                right_disp_map_path=osp.join(
                    dispDir, sceneDir, 'right', dispName
                ),
            )
            Metas.append(meta)
    return Metas


def getDrivingMetas(root, data_type='clean'):
    Metas = []

    imgDir = 'driving/frames_' + data_type + 'pass'
    dispDir = 'driving/disparity'

    focalLengthDirs = os.listdir(osp.join(root, dispDir))

    for focalLengthDir in focalLengthDirs:
        wardDirs = os.listdir(osp.join(root, dispDir, focalLengthDir))
        for wardDir in wardDirs:
            speedDirs = os.listdir(osp.join(root, dispDir, focalLengthDir, wardDir))
            for speedDir in speedDirs:
                dispNames = os.listdir(osp.join(root, dispDir, focalLengthDir, wardDir, speedDir, 'left'))
                imgNames = ["{}.png".format(name.split('.')[0]) for name in dispNames]
                for imgName, dispName in zip(imgNames, dispNames):
                    meta = dict(
                        left_image_path=osp.join(
                            imgDir, focalLengthDir, wardDir, speedDir, 'left', imgName
                        ),
                        right_image_path=osp.join(
                            imgDir, focalLengthDir, wardDir, speedDir, 'right', imgName
                        ),
                        left_disp_map_path=osp.join(
                            dispDir, focalLengthDir, wardDir, speedDir, 'left', dispName
                        ),
                        right_disp_map_path=osp.join(
                            dispDir, focalLengthDir, wardDir, speedDir, 'right', dispName
                        ),
                    )
                    Metas.append(meta)
    return Metas


def build_annoFile(root, save_annotation_root, data_type='clean'):
    """
    Build annotation files for Scene Flow Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    mkdir_or_exist(save_annotation_root)

    trainMetas = getFlying3dMetas(root, 'TRAIN', data_type)
    testMetas = getFlying3dMetas(root, 'TEST', data_type)

    trainMetas.extend(getMonkaaMetas(root, data_type))
    trainMetas.extend(getDrivingMetas(root, data_type))

    for meta in tqdm(trainMetas):
        for k, v in meta.items():
            assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)

    for meta in tqdm(testMetas):
        for k, v in meta.items():
            assert osp.exists(osp.join(root, v)), 'testMetas: {} not exists'.format(v)

    info_str = 'SceneFlow Dataset contains:\n' \
               '    {:5d}   training samples \n' \
               '    {:5d} validation samples'.format(len(trainMetas), len(testMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, data_type + 'pass_' + name + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    make_json(name='train', metas=trainMetas)
    make_json(name='test', metas=testMetas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SceneFlow Data PreProcess.")
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
        "--data-type",
        default='clean',
        help="the type of data, (clean or final)pass",
        type=str,
    )
    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.data_type)
