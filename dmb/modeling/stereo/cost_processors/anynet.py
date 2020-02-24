import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu, conv_bn_relu
from dmb.modeling.stereo.cost_processors.utils.hw_hourglass import HWHourglass
from .utils.dif_fms import fast_dif_fms
from .aggregators.anynet import AnyNetAggregator

class AnyNetProcessor(nn.Module):
    """
    An implementation of cost procession in DeepPruner

    Inputs:
        stage, (str): "pre"(Pre-PatchMatch) using patch match as sampler,
                   or "post"(Post-ConfidenceRangePredictor) using uniform sampler
        left, (tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        right, (tensor): Right image feature, in [BatchSize, Channels, Height, Width] layout
        disparity_sample, (tensor): The generated disparity samples for each pixel,
                           in [BatchSize, disparity_sample_number, Height, Width] layout
        min_disparity_feature, (tensor): the features used to estimate lower bound of disparity,
                  in [BatchSize, disparity_sample_number, Height, Width] layout
        max_disparity_feature, (tensor): the features used to estimate upper bound of disparity,
                  in [BatchSize, disparity_sample_number, Height, Width] layout

    Outputs:
        output, (tuple):
            For 'pre' stage, including:
            min_disparity, (tensor): the estimated lower bound of disparity,
                      in [BatchSize, 1, Height, Width] layout
            max_disparity, (tensor): the estimated upper bound of disparity,
                      in [BatchSize, 1, Height, Width] layout
            min_disparity_feature, (tensor): the features used to estimate lower bound of disparity,
                      in [BatchSize, patch_match_disparity_sample_number, Height, Width] layout
            max_disparity_feature, (tensor): the features used to estimate upper bound of disparity,
                      in [BatchSize, patch_match_disparity_sample_number, Height, Width] layout

            For 'post' stage, including:
            disparity, (tensor): the estimated disparity map,
                      in [BatchSize, 1, Height, Width] layout
            disparity_feature, (tensor): the features used to estimate aggregated disparity,
                      in [BatchSize, uniform_disparity_sample_number, Height, Width] layout

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

        # [B, D, H, W]
        cost = self.aggregator[stage](raw_cost)

        return cost




