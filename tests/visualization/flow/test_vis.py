import os
import sys
import torch
import torch.nn as nn
from thop import profile
from collections import Iterable
import time
import matplotlib.pyplot as plt
from imageio import imread
import unittest

from dmb.visualization.flow.vis import flow_to_color, flow_max_rad, flow_err_to_color, group_color
import dmb.data.datasets.utils.load_flow as Loader
from dmb.data.datasets.evaluation.flow.pixel_error import calc_error
from dmb.modeling.flow.layers.inverse_warp_flow import inverse_warp_flow


class testFlowVis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda:1')
        cls.root = '/home/youmin/data/bamboo_2/'
        cls.leftImagePath = os.path.join(cls.root, 'data/frame_0036.png')
        cls.rightImagePath = os.path.join(cls.root, 'data/frame_0037.png')
        cls.estFlowPath = os.path.join(cls.root, 'hmflow/frame_0036.flo')
        cls.gtFlowPath = os.path.join(cls.root, 'gt/frame_0036.flo')

        cls.estFlowSavePath = os.path.join(cls.root, 'test/frame_0036_est.png')
        cls.gtFlowSavePath = os.path.join(cls.root, 'test/frame_0036_gt.png')
        cls.errorFlowPath = os.path.join(cls.root, 'test/frame_0036_err.png')
        cls.groupFlowPath = os.path.join(cls.root, 'test/frame_0036_group.png')
        cls.warpRightImagePath = os.path.join(cls.root, 'test/frame_0036_left_warp.png')

        cls.estFlowWritePath = os.path.join(cls.root, 'test/frame_0036_est.flo')
        cls.gtFlowWritePath = os.path.join(cls.root, 'test/frame_0036_gt.flo')

        cls.estFlowWSPath = os.path.join(cls.root, 'test/frame_0036_ws.png')

        cls.leftImage = imread(cls.leftImagePath)
        cls.rightImage = imread(cls.rightImagePath)
        cls.estFlow = Loader.load_flying_chairs_flow(cls.estFlowPath)
        cls.gtFlow = Loader.load_flying_chairs_flow(cls.gtFlowPath)

    def readFlowSave(self, path, flow, max_rad=-1):
        max_rad = max(flow_max_rad(flow), max_rad)
        flowColor = flow_to_color(flow, max_rad)
        plt.imsave(path, flowColor, cmap=plt.cm.hot)

    unittest.skip('skip')
    def testColor(self):
        self.readFlowSave(self.estFlowSavePath, self.estFlow)
        self.readFlowSave(self.gtFlowSavePath, self.gtFlow)

    unittest.skip('skip')
    def testError(self):
        errorColor = flow_err_to_color(self.estFlow, self.gtFlow)
        plt.imsave(self.errorFlowPath, errorColor, cmap=plt.cm.hot)

    unittest.skip('skip')
    def testGroup(self):
        groupColor = group_color(self.estFlow, self.gtFlow, self.leftImage, self.rightImage,
                                 self.groupFlowPath)
    unittest.skip('skip')
    def testCalcError(self):
        error = calc_error(torch.Tensor(self.estFlow).permute(2, 0, 1),
                           torch.Tensor(self.gtFlow).permute(2, 0, 1), sparse=False)
        print('\nError\n')
        for k, v in error.items():
            print('{}: {}'.format(k, v))

    unittest.skip('skip')
    def testSave(self):
        Loader.write_flying_chairs_flow(self.estFlowWritePath, self.estFlow)
        self.estFlowWS = Loader.load_flying_chairs_flow(self.estFlowWritePath)
        self.readFlowSave(self.estFlowWSPath, self.estFlowWS)

    # unittest.skip('skip')
    def testWarp(self):
        flow  = torch.Tensor(self.gtFlow).permute(2, 0, 1).contiguous().unsqueeze(0)
        left  = torch.Tensor(self.leftImage).permute(2, 0, 1).contiguous().unsqueeze(0)
        right  = torch.Tensor(self.rightImage).permute(2, 0, 1).contiguous().unsqueeze(0)

        # [B, 3, H, W]
        warp_right = inverse_warp_flow(right, flow)
        warp_right = torch.cat((left, warp_right, right), dim=2)
        # [2H, W, 3]
        warp_right = warp_right[0, :, :, :].permute(1, 2, 0).contiguous().numpy() / 255.0
        warp_right = warp_right.clip(0., 1.)

        plt.imsave(self.warpRightImagePath, warp_right, cmap=plt.cm.hot)


if __name__ == '__main__':
    unittest.main()
