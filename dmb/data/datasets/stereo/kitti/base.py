import os.path as osp

import numpy as np
from imageio import imread

from dmb.data.datasets.stereo.base import StereoDatasetBase


class KittiDatasetBase(StereoDatasetBase):

    def __init__(self, annFile, root, transform=None, toRAM=False):
        super(KittiDatasetBase, self).__init__(annFile, root, transform)
        self.toRAM = toRAM
        self.listInRAM = None

        if self.toRAM:
            self.LoadToRAM()

    def LoadToRAM(self):
        self.listInRAM = {}
        for i in range(len(self.data_list)):
            key = self.data_list[i]['left_image_path']
            self.listInRAM.update({key: self.imageLoader(self.data_list[i])})

    def ImageLoader(self, item):
        # only take first three(RGB) channels, no matter in RGB or RGBA format
        leftImage = imread(
            osp.join(self.root, item['left_image_path'])
        )
        leftImage = leftImage.transpose(2, 0, 1).astype(np.float32)[:3]
        rightImage = imread(
            osp.join(self.root, item['right_image_path'])
        )
        rightImage = rightImage.transpose(2, 0, 1).astype(np.float32)[:3]

        h, w = leftImage.shape[1], leftImage.shape[2]
        original_size = (h, w)

        if 'left_disp_map_path' in item.keys() and item['left_disp_map_path'] is not None:
            leftDisp = imread(
                osp.join(self.root, item['left_disp_map_path'])
            ).astype(np.float32) / 256.0
            leftDisp = leftDisp[np.newaxis, ...]

        else:
            leftDisp = None

        if 'right_disp_map_path' in item.keys() and item['right_disp_map_path'] is not None:
            rightDisp = imread(
                osp.join(self.root, item['right_disp_map_path'])
            ).astype(np.float32) / 256.0
            rightDisp = rightDisp[np.newaxis, ...]

        else:
            rightDisp = None

        return {
            'leftImage': leftImage,
            'rightImage': rightImage,
            'leftDisp': leftDisp,
            'rightDisp': rightDisp,
            'original_size': original_size,
        }

    def Loader(self, item):
        if self.toRAM:
            return self.listInRAM[item['left_image_path']]
        else:
            return self.ImageLoader(item)

    @property
    def name(self):
        return 'KITTI'
