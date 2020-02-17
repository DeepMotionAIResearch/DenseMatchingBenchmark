import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu, conv_bn_relu
from dmb.modeling.stereo.cost_processors.utils.hw_hourglass import HWHourglass
from .utils.cat_fms import fast_cat_fms
from .aggregators import build_cost_aggregator


class ConfidenceRangePredictor(nn.Module):
    """
    Args:
        in_planes (int): the channels of raw cost volume
        hourglass_in_planes (int): the channels of hourglass module for cost aggregation
        disparity_sample_number, (int): the number of disparity samples
        batch_norm (bool): whether use batch normalization layer,
            default True

    Inputs:
        raw_cost, (tensor): the raw cost volume,
                  in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        disparity_sample, (tensor): the generated disparity samples,
                  in [BatchSize, disparity_sample_number, Height, Width] layout

    Outputs:
        min_disparity, (tensor): the estimated lower bound of disparity,
                  in [BatchSize, 1, Height, Width] layout
        max_disparity, (tensor): the estimated upper bound of disparity,
                  in [BatchSize, 1, Height, Width] layout
        min_disparity_feature, (tensor): the features used to estimate lower bound of disparity,
                  in [BatchSize, disparity_sample_number, Height, Width] layout
        max_disparity_feature, (tensor): the features used to estimate upper bound of disparity,
                  in [BatchSize, disparity_sample_number, Height, Width] layout

    """
    def __init__(self, in_planes, hourglass_in_planes, disparity_sample_number, batch_norm=True):
        super(ConfidenceRangePredictor, self).__init__()
        self.in_planes = in_planes
        self.hourglass_in_planes = hourglass_in_planes
        self.disparity_sample_number = disparity_sample_number
        self.batch_norm = batch_norm

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            conv3d_bn_relu(batch_norm, 64, 32, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv3d_bn_relu(batch_norm, 32, hourglass_in_planes, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.min_disparity_predictor = nn.Sequential(
            HWHourglass(hourglass_in_planes, batch_norm),
            conv3d_bn_relu(batch_norm, hourglass_in_planes, hourglass_in_planes*2,
                           kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv3d(hourglass_in_planes*2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.max_disparity_predictor = nn.Sequential(
            HWHourglass(hourglass_in_planes, batch_norm),
            conv3d_bn_relu(batch_norm, hourglass_in_planes, hourglass_in_planes*2,
                           kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv3d(hourglass_in_planes*2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # batch norm cannot be used here, as disparity map is the input and output
        self.min_disparity_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )

        # batch norm cannot be used here, as disparity map is the input and output
        self.max_disparity_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )

        self.min_disparity_feature_conv = conv_bn_relu(batch_norm, disparity_sample_number, disparity_sample_number,
                                                       kernel_size=5, stride=1, padding=2, dilation=1, bias=True)

        self.max_disparity_feature_conv = conv_bn_relu(batch_norm, disparity_sample_number, disparity_sample_number,
                                                       kernel_size=5, stride=1, padding=2, dilation=1, bias=True)

    def forward(self, raw_cost, disparity_sample):
        # in: [B, in_planes, D, H, W], out: [B, 64, D, H, W]
        cost = self.dres0(raw_cost)
        # in: [B, 64, D, H, W], out: [B, hourglass_in_planes, D, H, W]
        cost = self.dres1(cost)

        # in: [B, hourglass_in_planes, D, H, W], mid: [B, 1, D, H, W], out: [B, D, H, W]
        cost_for_min = self.min_disparity_predictor(cost).squeeze(1)

        # in: [B, hourglass_in_planes, D, H, W], mid: [B, 1, D, H, W], out: [B, D, H, W]
        cost_for_max = self.max_disparity_predictor(cost).squeeze(1)

        # soft arg-min
        # in: [B, D, H, W], out: [B, D, H, W]
        prob_for_min = F.softmax(cost_for_min, dim=1)
        # in: [B, D, H, W] * [B, D, H, W], out: [B, 1, H, W]
        min_disparity = torch.sum(prob_for_min * disparity_sample, dim=1, keepdim=True)
        # in: [B, 1, H, W], out: [B, 1, H, W]
        min_disparity = self.min_disparity_conv(min_disparity)

        # soft arg-min
        # in: [B, D, H, W], out: [B, D, H, W]
        prob_for_max = F.softmax(cost_for_max, dim=1)
        # in: [B, D, H, W] * [B, D, H, W], out: [B, 1, H, W]
        max_disparity = torch.sum(prob_for_max * disparity_sample, dim=1, keepdim=True)
        # in: [B, 1, H, W], out: [B, 1, H, W]
        max_disparity = self.max_disparity_conv(max_disparity)

        # in: [B, D, H, W], out: [B, D, H, W]
        min_disparity_feature = self.min_disparity_feature_conv(cost_for_min)
        # in: [B, D, H, W], out: [B, D, H, W]
        max_disparity_feature = self.max_disparity_feature_conv(cost_for_max)

        return min_disparity, max_disparity, min_disparity_feature, max_disparity_feature


class DeepPrunerProcessor(nn.Module):
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
        super(DeepPrunerProcessor, self).__init__()
        self.cfg = cfg.copy()
        self.batch_norm = cfg.model.batch_norm

        self.patch_match_disparity_sample_number = cfg.model.cost_processor.patch_match_disparity_sample_number
        self.uniform_disparity_sample_number = cfg.model.cost_processor.uniform_disparity_sample_number

        # setting confidence range predictor
        self.confidence_range_predictor_args = cfg.model.cost_processor.confidence_range_predictor
        self.confidence_range_predictor_args.update(
            # besides the disparity samples generated by PatchMatch, it also includes min, max disparity
            disparity_sample_number=self.patch_match_disparity_sample_number,
            batch_norm = self.batch_norm
        )
        self.confidence_range_predictor = ConfidenceRangePredictor(
            **self.confidence_range_predictor_args
        )

        # setting cost aggregator
        self.cost_aggregator = build_cost_aggregator(cfg)
        # batch norm cannot be used here, as disparity map is the input and output
        self.disparity_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.disparity_feature_conv = conv_bn_relu(self.batch_norm,
                                                   in_planes=self.uniform_disparity_sample_number,
                                                   out_planes=self.uniform_disparity_sample_number,
                                                   kernel_size=5, stride=1, padding=2,
                                                   dilation=1, bias=True)

    def forward(self, stage, left, right, disparity_sample, min_disparity_feature=None, max_disparity_feature=None):
        # [B, 2*C, D, H, W]
        raw_cost = fast_cat_fms(left, right, disp_sample=disparity_sample)

        # [B, 2*C+1, D, H, W]
        raw_cost = torch.cat((raw_cost, disparity_sample.unsqueeze(1)), dim=1)

        if stage=='pre': # (Pre-PatchMatch) using patch match as sampler,
            output = self.confidence_range_predictor(raw_cost, disparity_sample)

        else: # 'post', (Post-ConfidenceRangePredictor) using uniform sampler
            # [B, path_match_disparity_sample_number, H, W] ->
            # [B, path_match_disparity_sample_number, 1, H, W] ->
            # [B, path_match_disparity_sample_number, uniform_disparity_sample_number, H, W]
            min_disparity_feature = min_disparity_feature.unsqueeze(2).expand(-1, -1, self.uniform_disparity_sample_number, -1, -1)
            max_disparity_feature = max_disparity_feature.unsqueeze(2).expand(-1, -1, self.uniform_disparity_sample_number, -1, -1)

            # [B, 2*C+2*path_match_disparity_sample_number+1, uniform_disparity_sample_number, H, W]
            raw_cost = torch.cat((raw_cost, min_disparity_feature, max_disparity_feature), dim=1)

            # the returned cost after cost aggregation is in tuple
            # [B, uniform_disparity_sample_number, H, W]
            cost = self.cost_aggregator(raw_cost)[0]

            # soft arg-min
            # in: [B, D, H, W], out: [B, D, H, W]
            prob_volume = F.softmax(cost, dim=1)
            # in: [B, D, H, W] * [B, D, H, W], out: [B, 1, H, W]
            disparity = torch.sum(prob_volume * disparity_sample, dim=1, keepdim=True)

            # in: [B, 1, H, W], out: [B, 1, H*2, W*2]
            disparity = F.interpolate(disparity*2, scale_factor=(2,2), mode='bilinear', align_corners=False)

            # in: [B, D, H*2, W*2], out: [B, D, H*2, W*2]
            disparity_feature = F.interpolate(cost, scale_factor=(2,2), mode='bilinear', align_corners=False)

            # in: [B, 1, H*2, W*2], out: [B, 1, H*2, W*2]
            disparity = self.disparity_conv(disparity)

            # in: [B, D, H*2, W*2], out: [B, D, H*2, W*2]
            disparity_feature = self.disparity_feature_conv(disparity_feature)

            output = [disparity, disparity_feature]

        return output





