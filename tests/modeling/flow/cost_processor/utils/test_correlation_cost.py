import torch
import torch.nn as nn
import numpy as np

import unittest
import time

from dmb.modeling.flow.cost_processors.utils.correlation_cost import CorrelationCost


class TestCostComputation(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda:1')
        self.iters = 50


    # @unittest.skip("Just skipping")
    def test_correlation_cost(self):

        print('*' * 40, 'Test Correlation Cost', '*' * 40)

        H, W = 3, 4
        left = torch.linspace(1, H * W, H * W).reshape(1, 1, H, W).to(self.device)
        right = torch.linspace(H * W + 1, H * W * 2, H * W).reshape(1, 1, H, W).to(self.device)
        print('left: \n ', left)
        print('right: \n ', right)
        max_displacement = 2

        corr = CorrelationCost(max_displacement)
        cost = corr(left, right)
        print('Cost in shape: ', cost.shape)

        for i in range(cost.shape[1]):
            print('Channel {}:\n {}'.format(i, cost[:, i, ]))


    def timeTemplate(self, module, module_name, *args, **kwargs):
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        if isinstance(module, nn.Module):
            module.eval()

        start_time = time.time()

        for i in range(self.iters):
            with torch.no_grad():
                if len(args) > 0:
                    module(*args)
                if len(kwargs) > 0:
                    module(**kwargs)
                torch.cuda.synchronize(self.device)
        end_time = time.time()
        avg_time = (end_time - start_time) / self.iters
        print('{} reference forward once takes {:.4f}s, i.e. {:.2f}fps'.format(module_name, avg_time, (1 / avg_time)))

        if isinstance(module, nn.Module):
            module.train()

    # @unittest.skip("Just skipping")
    def test_speed(self):
        max_displacement = 4
        scale = 1
        SH, SW = 540, 960
        B, C, H, W = 1, 32, SH//scale, SW//scale

        reference_fm = torch.rand(B, C, H, W).to(self.device)
        target_fm = torch.rand(B, C, H, W).to(self.device)

        corr = CorrelationCost(max_displacement)
        self.timeTemplate(corr, 'CorrelationCost', reference_fm, target_fm)

if __name__ == '__main__':
    unittest.main()

'''

Test on GTX1080Ti, 540x960

'''
