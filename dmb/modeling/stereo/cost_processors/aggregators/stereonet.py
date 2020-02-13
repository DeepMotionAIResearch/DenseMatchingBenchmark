import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu


class StereoNetAggregator(nn.Module):
    """
    Args:
        max_disp (int): max disparity
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): difference-based cost volume without further processing,
            in [BatchSize, in_planes, max_disp//8, Height//8, Width//8] layout (default)
            or in [BatchSize, in_planes, max_disp//16, Height//16, Width//16] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, max_disp//8, Height//8, Width//8] layout (default)
            or in [BatchSize, in_planes, max_disp//16, Height//16, Width//16] layout
    """

    def __init__(self, max_disp, in_planes=32, batch_norm=True, num=4):
        super(StereoNetAggregator, self).__init__()
        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm
        self.num = num

        self.classify = nn.ModuleList([
            conv3d_bn_relu(self.batch_norm, in_planes, 32, kernel_size=3,
                           stride=1, padding=1, dilation=1, bias=True) for _ in range(self.num)
        ])

        self.lastconv = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, raw_cost):
        # default down-sample to 1/8 resolution, it also can be 1/16
        # raw_cost: (BatchSize, Channels, MaxDisparity/8, Height/8, Width/8)
        for i in range(self.num):
            raw_cost = self.classify[i](raw_cost)

        # cost: (BatchSize, 1, MaxDisparity/8, Height/8, Width/8)
        cost = self.lastconv(raw_cost)

        # (BatchSize, MaxDisparity/8, Height/8, Width/8)
        cost = torch.squeeze(cost, 1)


        return [cost]
