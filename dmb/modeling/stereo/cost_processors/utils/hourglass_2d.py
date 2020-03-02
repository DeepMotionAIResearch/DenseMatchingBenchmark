import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn, conv_bn_relu, deconv_bn


class Hourglass2D(nn.Module):
    """
    An implementation of 2d hourglass module proposed in PSMNet.
    Args:
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer,
            default True
    Inputs:
        x, (Tensor): cost volume
            in [BatchSize, in_planes, Height, Width] layout
        presqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, Height/2, Width/2] layout
        postsqu, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, Height/2, Width/2] layout
    Outputs:
        out, (Tensor): cost volume
            in [BatchSize, in_planes, MaxDisparity, Height, Width] layout
        pre, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, Height/2, Width/2] layout
        post, (optional, Tensor): cost volume
            in [BatchSize, in_planes * 2, Height/2, Width/2] layout

    """
    def __init__(self, in_planes, batch_norm=True):
        super(Hourglass2D, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = conv_bn_relu(
            self.batch_norm, in_planes, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )

        self.conv2 = conv_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv3 = conv_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv4 = conv_bn_relu(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv5 = deconv_bn(
            self.batch_norm, in_planes * 2, in_planes * 2,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )
        self.conv6 = deconv_bn(
            self.batch_norm, in_planes * 2, in_planes,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

    def forward(self, x, presqu=None, postsqu=None):
        # in: [B, C, H, W], out: [B, 2C, H/2, W/2]
        out = self.conv1(x)
        # in: [B, 2C, H/2, W/2], out: [B, 2C, H/2, W/2]
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # in: [B, 2C, H/2, W/2], out: [B, 2C, H/4, W/4]
        out = self.conv3(pre)
        # in: [B, 2C, H/4, W/4], out: [B, 2C, H/4, W/4]
        out = self.conv4(out)

        # in: [B, 2C, H/4, W/4], out: [B, 2C, H/2, W/2]
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        # in: [B, 2C, H/2, W/2], out: [B, C, H, W]
        out = self.conv6(post)

        return out, pre, post
