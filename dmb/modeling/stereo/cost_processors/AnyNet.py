import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.dif_fms import fast_dif_fms
from .aggregators.AnyNet import AnyNetAggregator

class AnyNetProcessor(nn.Module):
    """
    An implementation of cost procession in AnyNet

    Inputs:
        stage, (str): 'init_guess', the coarsest disparity estimation,
                      'warp_level_8', refine the disparity estimation with feature warp at resolution=1/8
                      'warp_level_4', refine the disparity estimation with feature warp at resolution=1/4
        left, (tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        right, (tensor): Right image feature, in [BatchSize, Channels, Height, Width] layout
        disp, (tensor): Disparity map outputted from last stage, in [BatchSize, 1, Height, Width] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout

    """

    def __init__(self, cfg):
        super(AnyNetProcessor, self).__init__()
        self.cfg = cfg.copy()
        self.batch_norm = cfg.model.batch_norm

        self.stage = self.cfg.model.stage

        # cost computation parameters, dict
        self.max_disp = self.cfg.model.cost_processor.cost_computation.max_disp
        self.start_disp = self.cfg.model.cost_processor.cost_computation.start_disp
        self.dilation = self.cfg.model.cost_processor.cost_computation.dilation


        # cost aggregation
        self.aggregator_type = self.cfg.model.cost_processor.cost_aggregator.type
        self.aggregator = nn.ModuleDict()
        for st in self.stage:
            self.aggregator[st] = AnyNetAggregator(
                in_planes=self.cfg.model.cost_processor.cost_aggregator.in_planes[st],
                agg_planes=self.cfg.model.cost_processor.cost_aggregator.agg_planes[st],
                num=self.cfg.model.cost_processor.cost_aggregator.num,
                batch_norm=self.batch_norm,
            )

    def forward(self, stage, left, right, disp=None):
        B, C, H, W = left.shape
        # construct the raw cost volume

        end_disp = self.start_disp[stage] + self.max_disp[stage] - 1

        # disparity sample number
        D = (self.max_disp[stage] + self.dilation[stage] - 1) // self.dilation[stage]

        # generate disparity samples, in [B, D, H, W] layout
        disp_sample = torch.linspace(self.start_disp[stage], end_disp, D)
        disp_sample = disp_sample.view(1, D, 1, 1).expand(B, D, H, W).to(left.device).float()

        # if initial disparity guessed, used for warping
        if disp is not None:
            # up-sample disparity map to the size of left
            H, W = left.shape[-2:]
            scale = W / disp.shape[-1]
            disp = F.interpolate(disp * scale, size=(H, W), mode='bilinear', align_corners=False)
            # shift the disparity sample to be centered at the given disparity map
            disp_sample = disp_sample + disp

        # [B, C, D, H, W]
        raw_cost = fast_dif_fms(left, right, disp_sample=disp_sample)

        # list [[B, D, H, W]]
        cost = self.aggregator[stage](raw_cost)

        return cost




