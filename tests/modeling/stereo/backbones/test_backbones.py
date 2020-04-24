import os
import sys
import torch
import torch.nn as nn
from thop import profile
from collections import Iterable
import time
import unittest

from dmb.modeling.stereo.backbones import build_backbone
from mmcv import Config



class testBackbones(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda:1')
        config_path = '/home/zhixiang/youmin/projects/depth/public/' \
                      'DenseMatchingBenchmark/configs/AcfNet/scene_flow_uniform.py'
        cls.cfg = Config.fromfile(config_path)
        cls.backbone = build_backbone(cls.cfg)
        cls.backbone.to(cls.device)

        cls.setUpTimeTestingClass()
        cls.avg_time = {}

    @classmethod
    def setUpTimeTestingClass(cls):
        cls.iters = 50

        h, w = 384, 1248
        leftImage = torch.rand(1, 3, h, w).to(cls.device)
        rightImage = torch.rand(1, 3, h, w).to(cls.device)

        cls.backbone_input = [leftImage, rightImage]

        print('Input preparation successful!')

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
        print('{} reference forward once takes {:.4f}ms, i.e. {:.2f}fps'.format(module_name, avg_time*1000, (1 / avg_time)))

        if isinstance(module, nn.Module):
            module.train()

        self.avg_time[module_name] = avg_time

    # @unittest.skip("demonstrating skipping")
    def test_0_OutputModel(self):
        print(self.backbone)

    # @unittest.skip("demonstrating skipping")
    def test_1_ModelTime(self):
        self.timeTemplate(self.backbone, 'Model', *self.backbone_input)


if __name__ == '__main__':
    unittest.main()


