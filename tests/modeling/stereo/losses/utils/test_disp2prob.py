import torch
import torch.nn as nn
from mmcv import Config
import time
import unittest

from dmb.modeling.stereo.losses.utils.disp2prob import LaplaceDisp2Prob, GaussianDisp2Prob, OneHotDisp2Prob


class testLosses(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:1")

    def testCase1Laplace(self):
        max_disp = 5
        start_disp = -2
        dilation = 2
        disp_sample=None
        variance = 2
        h, w = 3, 4

        gtDisp = torch.rand(1, 1, h, w) * max_disp + start_disp

        gtDisp = gtDisp.to(self.device)
        gtDisp.requires_grad = True
        print('*' * 60)
        print('Ground Truth Disparity:')
        print(gtDisp)


        print('*' * 60)
        print('Generated disparity probability volume:')
        prob_volume = LaplaceDisp2Prob(
                gtDisp, max_disp=max_disp, variance=variance,
                start_disp=start_disp, dilation=dilation, disp_sample=disp_sample
            ).getProb()

        idx = 0
        for i in range(start_disp, max_disp + start_disp, dilation):
            print('Disparity {}:\n {}'.format(i, prob_volume[:, idx, ]))
            idx += 1

    def testCase2Laplace(self):
        max_disp = 5
        start_disp = -2
        variance = 2
        h, w = 3, 4
        disp_sample = torch.Tensor([-2, 0, 2]).repeat(1, h, w, 1).permute(0, 3, 1, 2).contiguous()


        gtDisp = torch.rand(1, 1, h, w) * max_disp + start_disp

        gtDisp = gtDisp.to(self.device)
        gtDisp.requires_grad = True
        print('*' * 60)
        print('Ground Truth Disparity:')
        print(gtDisp)

        print('*' * 60)
        print('Generated disparity probability volume:')
        prob_volume = LaplaceDisp2Prob(
            gtDisp, max_disp=max_disp, start_disp=start_disp, variance=variance, disp_sample=disp_sample
        ).getProb()

        idx = 0
        for i in range(disp_sample.shape[1]):
            print('Disparity {}:\n {}'.format(i, prob_volume[:, idx, ]))
            idx += 1

if __name__ == '__main__':
    print('test probability volume!')
    unittest.main()

