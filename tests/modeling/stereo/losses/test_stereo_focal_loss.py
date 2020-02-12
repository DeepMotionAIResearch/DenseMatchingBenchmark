import torch
import torch.nn as nn
from mmcv import Config
import time
import unittest

from dmb.modeling.stereo.losses.builder import make_gsm_loss_evaluator, \
    make_focal_loss_evaluator, make_sll_loss_evaluator


class testLosses(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:1")

    # @unittest.skip("Just skipping")
    def testCase1(self):
        max_disp = 5
        start_disp = -2
        dilation = 2
        h, w = 3, 4
        d = (max_disp + dilation - 1) // dilation
        variance = 2

        gtDisp = torch.rand(1, 1, h, w) * max_disp + start_disp

        gtDisp = gtDisp.to(self.device)

        cfg = Config(dict(
            data = dict(
              sparse = False,
            ),
            model=dict(
                losses=dict(
                    focal_loss=dict(
                        # the maximum disparity of disparity search range
                        max_disp=max_disp,
                        # the start disparity of disparity search range
                        start_disp=start_disp,
                        # the step between near disparity sample
                        dilation=dilation,
                        # weight for stereo focal loss with regard to other loss type
                        weight=1.0,
                        # weights for different scale loss
                        weights=(1.0),
                        # stereo focal loss focal coefficient
                        coefficient=5.0,
                    )
                ),
            )
        ))

        estCost = torch.ones(1, d, h, w).to(self.device)
        estCost.requires_grad = True
        print('\n \n Test Case 1:')
        print('*' * 60)
        print('Estimated Cost volume:')
        print(estCost)

        stereo_focal_loss_evaluator = make_focal_loss_evaluator(cfg)
        print(stereo_focal_loss_evaluator)
        print('*' * 60)
        print(stereo_focal_loss_evaluator(estCost=estCost, gtDisp=gtDisp, variance=variance, disp_sample=None))

    def testCase2(self):
        max_disp = 5
        start_disp = -2
        variance = 2
        h, w = 3, 4
        disp_sample = torch.Tensor([-2, 0, 2]).repeat(1, h, w, 1).permute(0, 3, 1, 2).contiguous()

        d = disp_sample.shape[1]

        gtDisp = torch.rand(1, 1, h, w) * max_disp + start_disp

        gtDisp = gtDisp.to(self.device)

        gtDisp.requires_grad = True

        cfg = Config(dict(
            data=dict(
                sparse=False,
            ),
            model=dict(
                losses=dict(
                    focal_loss=dict(
                        # the maximum disparity of disparity search range
                        max_disp=max_disp,
                        # the start disparity of disparity search range
                        start_disp=start_disp,
                        # the step between near disparity sample
                        # dilation=dilation,
                        # weight for stereo focal loss with regard to other loss type
                        weight=1.0,
                        # weights for different scale loss
                        weights=(1.0),
                        # stereo focal loss focal coefficient
                        coefficient=5.0,
                    )
                ),
            )
        ))

        print('\n \n Test Case 2:')
        print('*' * 60)
        print('Ground Truth Disparity:')
        print(gtDisp)

        estCost = torch.ones(1, d, h, w).to(self.device)
        estCost.requires_grad = True
        print('*' * 60)
        print('Estimated Cost volume:')
        print(estCost)

        stereo_focal_loss_evaluator = make_focal_loss_evaluator(cfg)
        print(stereo_focal_loss_evaluator)
        print('*' * 60)
        print(stereo_focal_loss_evaluator(estCost=estCost, gtDisp=gtDisp, variance=variance, disp_sample=disp_sample))


if __name__ == '__main__':
    print('test losses!')
    unittest.main()

