import torch
import torch.nn as nn

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu
from dmb.modeling.stereo.cost_processors.utils.hw_hourglass import HWHourglass


class DeepPrunerAggregator(nn.Module):
    """
    Args:
        in_planes (int): the channels of raw cost volume
        hourglass_in_planes (int): the channels of hourglass module for cost aggregation
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): raw cost volume, in [BatchSize, in_planes, MaxDisparity, Height, Width] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout
    """

    def __init__(self, in_planes, hourglass_in_planes, batch_norm=True):
        super(DeepPrunerAggregator, self).__init__()
        self.in_planes = in_planes
        self.hourglass_in_planes = hourglass_in_planes
        self.batch_norm = batch_norm

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, in_planes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            conv3d_bn_relu(batch_norm, 64, 32, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv3d_bn_relu(batch_norm, 32, hourglass_in_planes, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.dres2 = HWHourglass(hourglass_in_planes, batch_norm=batch_norm)

        self.classify = nn.Sequential(
            conv3d_bn_relu(batch_norm, hourglass_in_planes, hourglass_in_planes * 2,
                           kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv3d(hourglass_in_planes * 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, raw_cost):
        # in: [B, in_planes, D, H, W], out: [B, 64, D, H, W]
        cost = self.dres0(raw_cost)
        # in: [B, 64, D, H, W], out: [B, hourglass_in_planes, D, H, W]
        cost = self.dres1(cost)

        # in: [B, hourglass_in_planes, D, H, W], out: [B, hourglass_in_planes, D, H, W]
        cost = self.dres2(cost) + cost

        # in: [B, hourglass_in_planes, D, H, W], mid: [B, 1, D, H, W], out: [B, D, H, W]
        cost = self.classify(cost).squeeze(1)

        return [cost]
