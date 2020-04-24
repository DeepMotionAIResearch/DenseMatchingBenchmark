import torch
import torch.nn as nn

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu, conv_bn_relu, conv_bn
from dmb.modeling.stereo.cost_processors.utils.hourglass import Hourglass


class AcfAggregator(nn.Module):
    """
    Args:
        max_disp (int): max disparity
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        raw_cost (Tensor): raw cost volume,
                in [BatchSize, Channels, MaxDisparity//4, Height//4, Width//4] layout

    Outputs:
        cost_volume (tuple of Tensor): cost volume
            in [BatchSize, MaxDisparity, Height, Width] layout
    """

    def __init__(self, max_disp, in_planes=64, batch_norm=True):
        super(AcfAggregator, self).__init__()
        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.dres0 = nn.Sequential(
            conv3d_bn_relu(batch_norm, self.in_planes, 32, 3, 1, 1),
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1),
        )
        self.dres1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1),
            conv3d_bn(batch_norm, 32, 32, 3, 1, 1)
        )
        self.dres2 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres3 = Hourglass(in_planes=32, batch_norm=batch_norm)
        self.dres4 = Hourglass(in_planes=32, batch_norm=batch_norm)

        self.classif1 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif2 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.classif3 = nn.Sequential(
            conv3d_bn_relu(batch_norm, 32, 32, 3, 1, 1),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.deconv1 = nn.ConvTranspose3d(1, 1, 8, 4, 2, bias=False)
        self.deconv2 = nn.ConvTranspose3d(1, 1, 8, 4, 2, bias=False)
        self.deconv3 = nn.ConvTranspose3d(1, 1, 8, 4, 2, bias=False)

    def forward(self, raw_cost):
        B, C, D, H, W = raw_cost.shape
        # concat_fms: (BatchSize, Channels*2, MaxDisparity/4, Height/4, Width/4)
        cost0 = self.dres0(raw_cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre2, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        # (BatchSize, 1, MaxDisparity, Height, Width)
        full_h, full_w = H * 4, W * 4

        cost1 = self.deconv1(cost1, [self.max_disp, full_h, full_w])
        cost2 = self.deconv2(cost2, [self.max_disp, full_h, full_w])
        cost3 = self.deconv3(cost3, [self.max_disp, full_h, full_w])

        # (BatchSize, MaxDisparity, Height, Width)
        cost1 = torch.squeeze(cost1, 1)
        cost2 = torch.squeeze(cost2, 1)
        cost3 = torch.squeeze(cost3, 1)

        return [cost3, cost2, cost1]


