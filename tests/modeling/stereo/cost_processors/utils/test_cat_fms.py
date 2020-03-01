import torch
import torch.nn as nn
import numpy as np

import unittest
import time

from dmb.modeling.stereo.cost_processors.utils.cat_fms import fast_cat_fms, cat_fms


test_dif_fms = False
if test_dif_fms:
    from dmb.modeling.stereo.cost_processors.utils.dif_fms import fast_dif_fms, dif_fms
    fast_cat_fms, cat_fms = fast_dif_fms, dif_fms



class TestCostComputation(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda:1')
        self.iters = 50


    # @unittest.skip("Just skipping")
    def test_cat_fms(self):

        print('*' * 40, 'Test cat_fms', '*' * 40)

        H, W = 3, 4
        left = torch.linspace(1, H * W, H * W).reshape(1, 1, H, W).to(self.device)
        right = torch.linspace(H * W + 1, H * W * 2, H * W).reshape(1, 1, H, W).to(self.device)
        print('left: \n ', left)
        print('right: \n ', right)
        start_disp = -2
        max_disp = 5
        dilation = 2
        d = (max_disp + dilation - 1) // dilation

        cost = cat_fms(left, right, max_disp, start_disp, dilation)
        print('Cost in shape: ', cost.shape)
        idx = 0
        for i in range(start_disp, max_disp + start_disp, dilation):
            print('Disparity {}:\n {}'.format(i, cost[:, :, idx, ]))
            idx += 1

        for i in range(cost.shape[1]):
            print('Channel {}:\n {}'.format(i, cost[:, i, ]))

        print('*' * 80)
        print('Test fast_cat_fms')

        cost = fast_cat_fms(left, right, max_disp, start_disp, dilation)
        print('Cost in shape: ', cost.shape)
        idx = 0
        for i in range(start_disp, max_disp + start_disp, dilation):
            print('Disparity {}:\n {}'.format(i, cost[:, :, idx, ]))
            idx += 1

        for i in range(cost.shape[1]):
            print('Channel {}:\n {}'.format(i, cost[:, i, ]))

        print('*' * 80)
        print('Test fast_cat_fms with disparity samples')

        end_disp = start_disp + max_disp - 1

        # generate disparity samples
        disp_samples = torch.linspace(start_disp, end_disp, d).repeat(1, H, W, 1). \
            permute(0, 3, 1, 2).contiguous().to(self.device)

        cost = fast_cat_fms(left, right, max_disp, start_disp, dilation, disp_samples)
        print('Cost in shape: ', cost.shape)
        idx = 0
        for i in range(start_disp, max_disp + start_disp, dilation):
            print('Disparity {}:\n {}'.format(i, cost[:, :, idx, ]))
            idx += 1

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
        max_disp = 192
        scale = 4
        start_disp = 0
        dilation = 1
        SH, SW = 540, 960
        B, C, H, W = 1, 32, SH//scale, SW//scale

        reference_fm = torch.rand(B, C, H, W).to(self.device)
        target_fm = torch.rand(B, C, H, W).to(self.device)

        self.timeTemplate(cat_fms, 'CAT_FMS', reference_fm, target_fm, max_disp//scale, start_disp, dilation)

        self.timeTemplate(fast_cat_fms, 'FAST_CAT_FMS', reference_fm, target_fm, max_disp//scale, start_disp, dilation)

        print('Test fast_cat_fms with disparity samples')

        d = (max_disp + dilation - 1) // dilation
        end_disp = start_disp + max_disp - 1

        # generate disparity samples
        disp_samples = torch.linspace(start_disp, end_disp, d).repeat(1, H, W, 1). \
            permute(0, 3, 1, 2).contiguous().to(self.device)

        self.timeTemplate(fast_cat_fms, 'FAST_CAT_FMS', reference_fm, target_fm, max_disp//scale, start_disp, dilation, disp_samples)


if __name__ == '__main__':
    unittest.main()

'''

Test on GTX1080Ti, 540x960

CAT_FMS reference forward once takes 0.2021s, i.e. 4.95fps
FAST_CAT_FMS reference forward once takes 0.0292s, i.e. 34.27fps

if directly provide disparity samples:
FAST_CAT_FMS reference forward once takes 0.0596s, i.e. 16.78fps

'''