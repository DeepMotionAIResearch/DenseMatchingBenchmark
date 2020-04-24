import torch
import torch.nn as nn
from mmcv import Config
import time
import unittest

from dmb.modeling.stereo.disp_predictors import build_disp_predictor


def kick_out_none_keys(cfg):
    none_cfg = []
    for k, v in cfg.items():
        if v is None:
            none_cfg.append(k)
    for k in none_cfg:
        cfg.pop(k)

    return cfg

class testDispPredictors(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:1")

        # FasterSoftArgmin
        self.pred_type = 'FASTER'
        self.radius = None
        self.radius_dilation = None

        # LocalSoftArgmin
        # self.pred_type = 'LOCAL'
        # self.radius = 2
        # self.radius_dilation = 1

        # SoftArgmin
        # self.pred_type = 'DEFAULT'
        # self.radius = None
        # self.radius_dilation = None

        self.iters = 50 # used for speed test

    def testCase1(self):
        start_disp = -4
        dilation = 2
        alpha = 1.0
        normalize = True
        max_disp = 9
        h, w = 2, 2

        d = (max_disp + dilation - 1) // dilation

        cfg = Config(dict(
            model=dict(
                disp_predictor=dict(
                    type=self.pred_type,
                    # the maximum disparity of disparity search range
                    max_disp=max_disp,
                    # disparity sample radius
                    radius=self.radius,
                    # the start disparity of disparity search range
                    start_disp=start_disp,
                    # the step between near disparity sample
                    dilation=dilation,
                    # the step between near disparity sample when local sampling
                    radius_dilation = self.radius_dilation,
                    # the temperature coefficient of soft argmin
                    alpha=alpha,
                    # whether normalize the estimated cost volume
                    normalize=normalize,

                ),
            )
        ))

        cfg.model.update(disp_predictor = kick_out_none_keys(cfg.model.disp_predictor))

        cost = torch.ones(1, d, h, w).to(self.device)
        cost.requires_grad = True
        print('*' * 60)
        print('Cost volume:')
        print(cost)

        disp_predictor = build_disp_predictor(cfg).to(self.device)
        print(disp_predictor)
        disp = disp_predictor(cost)
        print('*' * 60)
        print('Regressed disparity map :')
        print(disp)

        # soft argmin
        if self.pred_type == 'DEFAULT':
            print('*' * 60)
            print('Test directly providing disparity samples')

            end_disp = start_disp + max_disp - 1

            # generate disparity samples
            disp_samples = torch.linspace(start_disp, end_disp, d).repeat(1, h, w, 1).\
                                                permute(0, 3, 1, 2).contiguous().to(cost.device)
            disp = disp_predictor(cost, disp_samples)
            print('Regressed disparity map :')
            print(disp)

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

    # @unittest.skip('just want to skip')
    def testSpeed(self):
        start_disp = 0
        dilation = 1
        alpha = 1.0
        normalize = True
        max_disp = 192
        h, w = 544, 960
        d = (max_disp + dilation - 1) // dilation

        cfg = Config(dict(
            model=dict(
                disp_predictor=dict(
                    type=self.pred_type,
                    # the maximum disparity of disparity search range
                    max_disp=max_disp,
                    # disparity sample radius
                    radius=self.radius,
                    # the start disparity of disparity search range
                    start_disp=start_disp,
                    # the step between near disparity sample
                    dilation=dilation,
                    # the step between near disparity sample when local sampling
                    radius_dilation=self.radius_dilation,
                    # the temperature coefficient of soft argmin
                    alpha=alpha,
                    # whether normalize the estimated cost volume
                    normalize=normalize,

                ),
            )
        ))

        cfg.model.update(disp_predictor = kick_out_none_keys(cfg.model.disp_predictor))

        cost = torch.ones(1, d, h, w).to(self.device)
        cost.requires_grad = True
        print('*' * 60)
        print('Speed Test!')

        disp_predictor = build_disp_predictor(cfg).to(self.device)
        print(disp_predictor)
        self.timeTemplate(disp_predictor, self.pred_type, cost_volume=cost)

        # soft argmin
        if self.pred_type == 'DEFAULT':
            print('*' * 60)
            print('Speed test directly providing disparity samples')

            end_disp = start_disp + max_disp - 1

            # generate disparity samples
            disp_samples = torch.linspace(start_disp, end_disp, d).repeat(1, h, w, 1). \
                permute(0, 3, 1, 2).contiguous().to(cost.device)
            self.timeTemplate(disp_predictor, self.pred_type, cost_volume=cost, disp_sample=disp_samples)


if __name__ == '__main__':
    print('test disp predictors!')
    unittest.main()

"""
Speed Test on GTX1080Ti

SoftArgmin reference forward once takes 0.6142s, i.e. 1.63fps
FasterSoftArgmin reference forward once takes 0.0381s, i.e. 26.27fps
LocalSoftArgmin reference forward once takes 0.0032s, i.e. 308.52fps

if directly provide disparity samples:
SoftArgmin reference forward once takes 0.6142s, i.e. 76.91fps

"""