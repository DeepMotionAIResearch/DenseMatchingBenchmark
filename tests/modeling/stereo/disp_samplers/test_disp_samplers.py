import torch
import torch.nn as nn
import numpy as np
from mmcv import Config

import unittest
import time

from dmb.modeling.stereo.disp_samplers import build_disp_sampler


class TestDispSamplers(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda:6')
        self.iters = 50

        self.sampler_type = 'MONOSTEREO'


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
        SH, SW = 384, 1248
        B, C, H, W = 1, 8, SH//scale, SW//scale

        left = torch.rand(B, 2*C, H, W).to(self.device)
        right = torch.rand(B, 2*C, H, W).to(self.device)

        cfg = Config(dict(
            model=dict(
                batch_norm=True,
                disp_sampler=dict(
                    type=self.sampler_type,
                    # max disparity
                    max_disp=max_disp,
                    # the down-sample scale of the input feature map
                    scale=scale,
                    # the number of diaparity samples
                    disparity_sample_number=8,
                    # the in planes of extracted feature
                    in_planes=2*C,
                    # the base channels of convolution layer in this network
                    C=C,
                ),
            )
        ))

        disp_sampler = build_disp_sampler(cfg).to(self.device)

        print('*' * 60)
        print('Speed Test!')

        print('*' * 60)
        print('Correlation Speed!')
        self.timeTemplate(disp_sampler.correlation, 'Correlation', left, right)

        raw_cost = disp_sampler.correlation(left, right, max_disp//scale)

        print('*' * 60)
        print('Diaprity proposal aggregator Speed!')
        self.timeTemplate(disp_sampler.proposal_aggregator, 'Aggregator', raw_cost)

        proposal_disp, proposal_cost = disp_sampler.proposal_aggregator(raw_cost)

        print('*' * 60)
        print('Diaprity proposal sampler Speed!')
        context = torch.cat((proposal_disp, proposal_cost, left), dim=1)
        self.timeTemplate(disp_sampler.deformable_sampler, 'DeformableSampler', context, proposal_disp, None)

        print('*' * 60)
        print('Wholistic Module Speed!')
        self.timeTemplate(disp_sampler, self.sampler_type, left, right, proposal_disp, proposal_cost, None)


if __name__ == '__main__':
    unittest.main()

'''

Test on GTX1080Ti, 384x1248


Correlation reference forward once takes 0.1957s, i.e. 5.11fps
Aggregator reference forward once takes 0.0032s, i.e. 308.87fps
DeformableSampler reference forward once takes 0.0050s, i.e. 199.79fps
MONOSTEREO reference forward once takes 0.0064s, i.e. 155.05fps
'''