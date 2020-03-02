import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators.hm_net import HMAggregator
from .utils.correlation_cost import CorrelationCost
from dmb.modeling.flow.layers.inverse_warp_flow import inverse_warp_flow
from dmb.modeling.flow.flow_predictors import build_flow_predictor
from dmb.modeling.flow.flow_predictors.hm_net import HMPredictor

class HMNetProcessor(nn.Module):
    """
    An implementation of cost procession in HMNet

    Inputs:
        stage, (str): 'init_guess', the coarsest disparity estimation,
                      'warp_level_32', refine the disparity estimation with feature warp at resolution=1/32
                      'warp_level_16', refine the disparity estimation with feature warp at resolution=1/16
                      'warp_level_8', refine the disparity estimation with feature warp at resolution=1/8
                      'warp_level_4', refine the disparity estimation with feature warp at resolution=1/4
        left, (optional, tensor): Left image feature, in [BatchSize, Channels, Height, Width] layout
        right, (optional, tensor): Right image feature, in [BatchSize, Channels, Height, Width] layout
        global_cost, (optional, tensor): Cost volume including global information,
                               in [BatchSize, Channels, Height, Width] layout
        last_stage_cost, (optional, tensor): Cost volume of last stage in coarse-to-fine branch,
                               in [BatchSize, Channels, Height, Width] layout
        last_stage_flow, (optional, tensor): Flow map outputted from last stage, in [BatchSize, 2, Height, Width] layout

    Outputs:
        up_flow (tensor): Up-sampled flow map, in [BatchSize, 2, Height*2, Width*2]
        up_cost (tensor): Up-sampled cost volume in coarse-to-fine process, in [BatchSize, 2, Height*2, Width*2]

    """

    def __init__(self, cfg):
        super(HMNetProcessor, self).__init__()
        self.cfg = cfg.copy()
        self.batch_norm = cfg.model.batch_norm

        self.stage = self.cfg.model.stage

        # enable residual refinement for each level
        self.residual = self.cfg.model.cost_processor.residual

        # enable coarse-to-fine
        self.coarse_to_fine = self.cfg.model.cost_processor.coarse_to_fine

        # coarse-to-fine cost aggregation
        if self.coarse_to_fine:
            # cost computation parameters, dict
            self.max_displacement = self.cfg.model.cost_processor.cost_computation.max_displacement
            self.correlation = CorrelationCost(self.max_displacement)
    
            # cost aggregation
            self.aggregator_type = self.cfg.model.cost_processor.cost_aggregator.type
    
            # cost aggregation for coarse-to-fine
            self.aggregator = nn.ModuleDict()
            self.aggregator_in_planes = self.cfg.model.cost_processor.cost_aggregator.in_planes
            self.aggregator_out_planes = self.cfg.model.cost_processor.cost_aggregator.out_planes

            # set each stage during the coarse-to-fine process
            for st in self.stage:
                self.aggregator[st] = HMAggregator(
                    in_planes=self.aggregator_in_planes[st],
                    out_planes=self.aggregator_out_planes,
                    agg_planes_list=self.cfg.model.cost_processor.cost_aggregator.agg_planes_list,
                    dense=self.cfg.model.cost_processor.cost_aggregator.dense,
                    batch_norm=self.batch_norm,
                )

        # enable global matching
        self.global_matching = self.cfg.model.cost_processor.global_matching

        # global matching cost aggregation
        if self.global_matching:
            # cost aggregation for global information
            self.global_aggregator = nn.ModuleDict()
            self.global_flow_predictor = nn.ModuleDict()
            self.global_aggregator_in_planes = self.cfg.model.cost_processor.cost_aggregator.global_in_planes
            self.global_aggregator_out_planes = self.cfg.model.cost_processor.cost_aggregator.global_out_planes

            # set each stage during the global matching process
            for st in self.stage:
                self.global_aggregator[st] = nn.Sequential(
                    nn.Conv2d(self.global_aggregator_in_planes[st], self.global_aggregator_out_planes,
                              kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                    nn.LeakyReLU(0.1, inplace=True),
                )

                self.global_flow_predictor[st] = HMPredictor(in_planes=self.global_aggregator_out_planes,
                                                             batch_norm=self.batch_norm)


        # flow predictor
        self.flow_predictor = nn.ModuleDict()
        # up-sample predicted flow with 2x
        self.up_flow_op = nn.ModuleDict()
        # up-sample cost volume with 2x
        self.up_cost_op = nn.ModuleDict()

        # set each stage during the coarse-to-fine process
        for st in self.stage:
            self.flow_predictor[st] = build_flow_predictor(cfg)

            # up-sample predicted flow with 2x
            self.up_flow_op[st] = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=True)

            # if without coarse-to-fine process, cost only from global matching
            if self.coarse_to_fine:
                self.up_cost_in_planes = self.aggregator_out_planes
            else: # global matching
                self.up_cost_in_planes = self.global_aggregator_out_planes
            # up-sample cost volume of coarse-to-fine process with 2x
            self.up_cost_op[st] = nn.ConvTranspose2d(self.up_cost_in_planes, 2, kernel_size=4,
                                                     stride=2, padding=1, bias=True)

    def forward(self, stage, left=None, right=None, global_cost=None, last_stage_cost=None, last_stage_flow=None):
        # cost volume
        cost = None
        global_flow = None
        if self.global_matching:
            if last_stage_flow is not None:
                global_cost = torch.cat((global_cost, last_stage_cost, last_stage_flow), dim=1)
            # [B, global_aggregator_out_planes, H, W]
            cost = self.global_aggregator[stage](global_cost)

            global_flow = self.global_flow_predictor[stage](cost)

        if self.coarse_to_fine:
            if last_stage_flow is not None:
                right = inverse_warp_flow(right, last_stage_flow)
            # raw cost volume
            # [B, (max_displacement * 2 + 1)**2, H, W]
            raw_cost = self.correlation(left, right)

            # combine left image feature into correlation-based cost
            raw_cost = torch.cat((raw_cost, left), dim=1)

            # integrate global matching cost
            if self.global_matching:
                # combine global cost into correlation-based cost
                raw_cost = torch.cat((raw_cost, cost), dim=1)
    
            if last_stage_flow is not None:
                raw_cost = torch.cat((raw_cost, last_stage_flow, last_stage_cost), dim=1)
    
            # [B, aggregator_out_planes, H, W]
            cost = self.aggregator[stage](raw_cost)

        # [B, 2, H, W]
        flow = self.flow_predictor[stage](cost)

        # residual refinement
        # [B, 2, H, W]
        if self.residual and (last_stage_flow is not None):
            flow = flow + last_stage_flow

        # [B, 2, 2H, 2W], up-sample 2x, corresponding flow value x2
        up_flow = self.up_flow_op[stage](flow * 2)

        # [B, 2, 2H, 2W]
        up_cost = self.up_cost_op[stage](cost)


        return flow, cost, up_flow, up_cost, global_flow
