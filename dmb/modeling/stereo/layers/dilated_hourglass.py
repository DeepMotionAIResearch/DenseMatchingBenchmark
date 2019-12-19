import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn, conv3d_bn_relu, conv_bn_relu, deconv3d_bn


class Hourglass(nn.Module):
    def __init__(self, in_planes, batchNorm=True):
        super(Hourglass, self).__init__()
        self.batchNorm = batchNorm

        self.conv1 = conv3d_bn_relu(
            self.batchNorm, in_planes, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )

        self.conv2 = conv3d_bn(
            self.batchNorm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv3 = conv3d_bn_relu(
            self.batchNorm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv4 = conv3d_bn_relu(
            self.batchNorm, in_planes * 2, in_planes * 2,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv5 = deconv3d_bn(
            self.batchNorm, in_planes * 2, in_planes * 2,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )
        self.conv6 = deconv3d_bn(
            self.batchNorm, in_planes * 2, in_planes,
            kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

    def forward(self, x, presqu, postsqu):
        # in:1/4, out:1/8
        out = self.conv1(x)
        # in:1/8, out:1/8
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        # in:1/8, out:1/16
        out = self.conv3(pre)
        # in:1/16, out:1/16
        out = self.conv4(out)

        # in:1/16, out:1/8
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        # in:1/8, out:1/4
        out = self.conv6(post)

        return out, pre, post
