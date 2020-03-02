import torch
import torch.nn as nn

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu
from dmb.modeling.stereo.cost_processors.utils.hw_hourglass import HWHourglass


class MonoStereoAggregator(nn.Module):
    """
    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): raw cost volume, in [BatchSize, in_planes, MaxDisparity, Height, Width] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, 4*Height, 4*Width] layout
    """

    def __init__(self, in_planes, C, disparity_sample_number=3, batch_norm=True):
        super(MonoStereoAggregator, self).__init__()
        self.in_planes = in_planes
        self.C = C
        self.disparity_sample_number = disparity_sample_number
        self.batch_norm = batch_norm

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, in_planes, C*2, kernel_size=3, stride=1, padding=1, bias=False),
            conv3d_bn_relu(batch_norm, C*2, C*2, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, C*2, C*2, kernel_size=3, stride=1, padding=1, bias=False),
            conv3d_bn_relu(batch_norm, C*2, C*2, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.dres2 = HWHourglass(C*2, batch_norm=batch_norm)

        self.classify = nn.Sequential(
            conv3d_bn_relu(batch_norm, C*2, C*2,
                           kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv3d(C * 2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.deconv = nn.ConvTranspose2d(self.disparity_sample_number, self.disparity_sample_number, 8, 4, 2, bias=False)


    def forward(self, raw_cost):
        B, C, D, H, W = raw_cost.shape
        # in: [B, in_planes, D, H, W], out: [B, 2C, D, H, W]
        cost = self.dres0(raw_cost)
        # in: [B, 2C, D, H, W], out: [B, hourglass_in_planes, D, H, W]
        cost = self.dres1(cost)

        # in: [B, hourglass_in_planes, D, H, W], out: [B, hourglass_in_planes, D, H, W]
        cost = self.dres2(cost) + cost

        # in: [B, hourglass_in_planes, D, H, W], mid: [B, 1, D, H, W], out: [B, D, H, W]
        cost = self.classify(cost).squeeze(1)

        # (BatchSize, 1, MaxDisparity, Height, Width)
        full_h, full_w = H * 4, W * 4

        # in: [B, D, H, W], out: [B, D, 4H, 4W]
        cost = self.deconv(cost, [full_h, full_w])


        return [cost]
