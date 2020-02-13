import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu, conv_bn_relu, deconv3d_bn


class HWHourglass(nn.Module):
    """
    An implementation of hourglass module proposed in DeepPruner.
    Although input 3D cost volume, but stride is only imposed on Height, Width dimension

    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer,
            default True

    Inputs:
        raw_cost, (Tensor): raw cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout

    Outputs:
        cost, (Tensor): processed cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout

    """
    def __init__(self, in_planes, batch_norm=True):
        super(HWHourglass, self).__init__()
        self.batch_norm = batch_norm

        self.conv1_a = conv3d_bn_relu(
            self.batch_norm, in_planes, in_planes * 2,
            kernel_size=3, stride=(1,2,2), padding=1, bias=False
        )

        self.conv1_b = conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=(1,1,1), padding=1, bias=False
        )

        self.conv1_d = deconv3d_bn(
            self.batch_norm, in_planes * 2, in_planes,
            kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2), bias=False
        )

        self.conv2_a = conv3d_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 4,
            kernel_size=3, stride=(1, 2, 2), padding=1, bias=False
        )

        self.conv2_b = conv3d_bn_relu(
            self.batch_norm, in_planes * 4, in_planes * 4,
            kernel_size=3, stride=(1, 1, 1), padding=1, bias=False
        )

        self.conv2_d = deconv3d_bn(
            self.batch_norm, in_planes * 4, in_planes * 2,
            kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False
        )

        self.conv3_a = conv3d_bn_relu(
            self.batch_norm, in_planes * 4, in_planes * 8,
            kernel_size=3, stride=(1, 2, 2), padding=1, bias=False
        )

        self.conv3_b = conv3d_bn_relu(
            self.batch_norm, in_planes * 8, in_planes * 8,
            kernel_size=3, stride=(1, 1, 1), padding=1, bias=False
        )

        self.conv3_d = deconv3d_bn(
            self.batch_norm, in_planes * 8, in_planes * 4,
            kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False
        )


    def forward(self, raw_cost):
        # in: [B, C, D, H, W], out: [B, 2C, D, H/2, W/2]
        out1_a = self.conv1_a(raw_cost)

        # in: [B, 2C, D, H/2, W/2], out: [B, 2C, D, H/2, W/2]
        out1_b = self.conv1_b(out1_a) + out1_a

        # in: [B, 2C, D, H/2, W/2], out: [B, 4C, D, H/4, W/4]
        out2_a = self.conv2_a(out1_b)

        # in: [B, 4C, D, H/4, W/4], out: [B, 4C, D, H/4, W/4]
        out2_b = self.conv2_b(out2_a) + out2_a

        # in: [B, 8C, D, H/8, W/8], out: [B, 8C, D, H/8, W/8]
        out3_a = self.conv3_a(out2_b)

        # in: [B, 8C, D, H/8, W/8], out: [B, 8C, D, H/8, W/8]
        out3_b = self.conv3_b(out3_a) + out3_a

        # in: [B, 8C, D, H/8, W/8], out: [B, 4C, D, H/4, W/4]
        cost = self.conv3_d(out3_b) + out2_b

        # in: [B, 4C, D, H/4, W/4], out: [B, 2C, D, H/2, W/2]
        cost = self.conv2_d(cost) + out1_b

        # in: [B, 2C, D, H/2, W/2], out: [B, C, D, H, W]
        cost = self.conv1_d(cost)

        return cost
