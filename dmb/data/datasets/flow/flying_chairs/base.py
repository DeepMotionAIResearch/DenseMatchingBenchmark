import os.path as osp
import numpy as np
from imageio import imread

from dmb.data.datasets.flow.base import FlowDatasetBase
from dmb.data.datasets.utils import load_flying_chairs_flow


class FlyingChairsDataset(FlowDatasetBase):

    def __init__(self, annFile, root, transform=None):
        super(FlyingChairsDataset, self).__init__(annFile, root, transform)

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

        if 'flow_path' in item.keys() and item['flow_path'] is not None:
            flow = load_flying_chairs_flow(
                osp.join(self.root, item['flow_path'])
            ).transpose(2, 0, 1).astype(np.float32)

        else:
            flow = None


        return {
            'leftImage': leftImage,
            'rightImage': rightImage,
            'flow': flow,
            'original_size': original_size,
        }

    @property
    def name(self):
        return 'FlyingChairs'
