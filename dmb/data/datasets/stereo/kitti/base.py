import os.path as osp

import numpy as np
from imageio import imread

from dmb.data.datasets.stereo.base import StereoDatasetBase


class KittiDatasetBase(StereoDatasetBase):

    def __init__(self, annFile, root, transform=None):
        super(KittiDatasetBase, self).__init__(annFile, root, transform)

    def Loader(self, item):
        # only take first three RGB channel no matter in RGB or RGBA format
        leftImage = imread(
            osp.join(self.root, item['left_image_path'])
        ).transpose(2, 0, 1).astype(np.float32)[:3]
        rightImage = imread(
            osp.join(self.root, item['right_image_path'])
        ).transpose(2, 0, 1).astype(np.float32)[:3]

        h, w = leftImage.shape[1], leftImage.shape[2]
        original_size = (h, w)

        sample = {
            'leftImage': leftImage,
            'rightImage': rightImage,
            'original_size': original_size,
        }


        if 'left_disp_map_path' in item.keys() and item['left_disp_map_path'] is not None:
            leftDisp = imread(
                osp.join(self.root, item['left_disp_map_path'])
            ).astype(np.float32) / 256.0
            leftDisp = leftDisp[np.newaxis, ...]

            sample.update(leftDisp=leftDisp)

        if 'right_disp_map_path' in item.keys() and item['right_disp_map_path'] is not None:
            rightDisp = imread(
                osp.join(self.root, item['right_disp_map_path'])
            ).astype(np.float32) / 256.0
            rightDisp = rightDisp[np.newaxis, ...]

            sample.update(rightDisp=rightDisp)

        return sample

    @property
    def name(self):
        return 'KITTI'
