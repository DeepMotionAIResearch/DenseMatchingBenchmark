import torch
import torch.nn as nn
from mmcv import Config
import time
import unittest

from dmb.modeling.flow.losses import make_gof_loss_evaluator

class testLosses(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:1")

    # @unittest.skip("Just skipping")
    def testCase1(self):
        h, w = 3, 4
        cfg = Config(dict(
            data = dict(
              sparse = False,
            ),
            model=dict(
                losses=dict(
                    p_norm_loss=dict(
                        p=2.0,
                        epsilon=0.0,
                        # weight for stereo focal loss with regard to other loss type
                        weight=1.0,
                        # weights for different scale loss
                        weights=(1.0, 2.0),
                    )
                ),
            )
        ))

        gtFlow = torch.rand(1, 2, h, w)
        gtFlow = gtFlow.to(self.device)

        estFlow = torch.rand(1, 2, h, w).to(self.device)

        estFlow.requires_grad = True
        print('\n \n Test Case 1:')
        print('*' * 60)
        print('Estimated Flow:')
        print(estFlow)

        p_norm_loss_evaluator = make_gof_loss_evaluator(cfg)
        print(p_norm_loss_evaluator.loss_evaluators)
        print('*' * 60)
        print(p_norm_loss_evaluator(estFlow, gtFlow))


if __name__ == '__main__':
    print('test losses!')
    unittest.main()

