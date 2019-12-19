import os.path as osp
import numpy as np
from imageio import imread

from dmb.data.datasets.stereo.base import StereoDatasetBase
from dmb.data.datasets.utils import load_scene_flow_disp


class SceneFlowDataset(StereoDatasetBase):

    def __init__(self, annFile, root, transform=None):
        super(SceneFlowDataset, self).__init__(annFile, root, transform)

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

        if 'left_disp_map_path' in item.keys() and item['left_disp_map_path'] is not None:
            leftDisp = load_scene_flow_disp(
                osp.join(self.root, item['left_disp_map_path'])
            )
            leftDisp = leftDisp[np.newaxis, ...]

        else:
            leftDisp = None

        if 'right_disp_map_path' in item.keys() and item['right_disp_map_path'] is not None:
            rightDisp = load_scene_flow_disp(
                osp.join(self.root, item['right_disp_map_path'])
            )
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

    @property
    def name(self):
        return 'SceneFlow'
