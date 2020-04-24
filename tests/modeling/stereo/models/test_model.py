import os
import sys
import torch
import torch.nn as nn
from thop import profile
from collections import Iterable
import time
import unittest

from dmb.modeling import build_model
from mmcv import Config

def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums

def calcFlops(model, input):
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops: {} \nparameters: {}'.format(flops, params))
    return flops, params


class testModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda:2')
        config_path = '/home/zhixiang/youmin/projects/depth/public/' \
                      'DenseMatchingBenchmark/configs/PSMNet/kitti_2015.py'
        cls.cfg = Config.fromfile(config_path)
        cls.model = build_model(cls.cfg)
        cls.model.to(cls.device)

        cls.setUpTimeTestingClass()
        cls.avg_time = {}

    @classmethod
    def setUpTimeTestingClass(cls):
        cls.iters = 50

        h, w = 384, 1248
        leftImage = torch.rand(1, 3, h, w).to(cls.device)
        rightImage = torch.rand(1, 3, h, w).to(cls.device)
        leftDisp = torch.rand(1, 1, h, w).to(cls.device)
        batch = {'leftImage': leftImage,
                 'rightImage': rightImage,
                 'leftDisp': leftDisp, }

        cls.model_input = {
            'batch': batch
        }

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
    def test_2_OutputModel(self):
        print('\n', '*'*40, 'Model Configuration Result', '*'*40)
        print(self.model)
        calcFlops(self.model, self.model_input['batch'])

    # @unittest.skip("demonstrating skipping")
    def test_3_ModelTime(self):
        print('\n', '*'*40, 'Runtime Test Result', '*'*40)
        self.timeTemplate(self.model, 'Model', **self.model_input)

    # @unittest.skip("demonstrating skipping")
    def test_0_TrainingPhase(self):
        h, w = self.cfg.data.train.input_shape
        leftImage = torch.rand(1, 3, h, w).to(self.device)
        rightImage = torch.rand(1, 3, h, w).to(self.device)
        leftDisp = torch.rand(1, 1, h, w).to(self.device)
        batch = {'leftImage': leftImage,
                 'rightImage': rightImage,
                 'leftDisp': leftDisp,
                 }

        self.model.train()
        _, loss_dict = self.model(batch)

        print('\n', '*'*40, 'Train Result', '*'*40)
        for k, v in loss_dict.items():
            print(k, v)

        print(self.model.loss_evaluator.loss_evaluators)
        if hasattr(self.cfg.model, 'cmn'):
            print(self.model.cmn.loss_evaluator.loss_evaluators)

    # @unittest.skip("demonstrating skipping")
    def test_1_TestingPhase(self):
        h, w = self.cfg.data.test.input_shape
        leftImage = torch.rand(1, 3, h, w).to(self.device)
        rightImage = torch.rand(1, 3, h, w).to(self.device)
        leftDisp = torch.rand(1, 1, h, w).to(self.device)
        batch = {'leftImage': leftImage,
                 'rightImage': rightImage,
                 'leftDisp': leftDisp,
                 }

        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        self.model.eval()
        with torch.no_grad():
            result, _ = self.model(batch)

        print('\n', '*'*40, 'Test Result', '*'*40)
        print('Result for disparity:')
        print('Length of disparity map list: ', len(result['disps']))
        for i in range(len(result['disps'])):
            d = result['disps'][i]
            if d is not None and torch.is_tensor(d):
                print('Disparity {} with shape: '.format(i), d.shape)

        if 'costs' in result:
            print('Result for Cost: ')
            print('Length of cost list: ', len(result['costs']))
            if result['costs'][0] is not None:
                print(result['costs'][0].shape)
                print('Device of cost: ', result['costs'][0].device)

        if 'confs' in result:
            print('Result for Confidence map')
            print('Length of confidence list: ', len(result['confs']))
            for i in range(len(result['confs'])):
                conf = result['confs'][i]
                if conf is not None and torch.is_tensor(conf):
                    print('Confidence {} with shape: '.format(i), conf.shape)


if __name__ == '__main__':
    unittest.main()


