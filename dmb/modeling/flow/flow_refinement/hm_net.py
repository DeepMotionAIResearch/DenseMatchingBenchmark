import torch
import torch.nn as nn

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu


class HMRefinement(nn.Module):
    """
    Args:
        in_planes, (int): in channels of input
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        cost, (tensor): cost volume in [BatchSize, Channels, Height, Width] layout
        flow, (tensor): initial flow map in [BatchSize, 2, Height, Width] layout

    Outputs:
        refine_flow, (tensor): refined flow in [BatchSize, 2, Height, Width] layout
    """
    def __init__(self, in_planes, batch_norm=True):
        super(HMRefinement, self).__init__()

        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv0 = conv_bn_relu(batch_norm, in_planes, 128, kernel_size=3, stride=1,
                                  padding=1, dilation=1, bias=True)
        self.conv1 = conv_bn_relu(batch_norm,       128, 128, kernel_size=3, stride=1,
                                  padding=2, dilation=2, bias=True)
        self.conv2 = conv_bn_relu(batch_norm,       128, 128, kernel_size=3, stride=1,
                                  padding=4, dilation=4, bias=True)
        self.conv3 = conv_bn_relu(batch_norm,       128,  96, kernel_size=3, stride=1,
                                  padding=8, dilation=8, bias=True)
        self.conv4 = conv_bn_relu(batch_norm,        96,  64, kernel_size=3, stride=1,
                                  padding=16, dilation=16, bias=True)
        self.conv5 = conv_bn_relu(batch_norm,        64,  32, kernel_size=3, stride=1,
                                  padding=1, dilation=1, bias=True)

        self.predictor = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def forward(self, cost, flow):
        cost = self.conv0(cost)
        cost = self.conv1(cost)
        cost = self.conv2(cost)
        cost = self.conv3(cost)
        cost = self.conv4(cost)
        cost = self.conv5(cost)
        res_flow = self.predictor(cost)
        refine_flow = flow + res_flow

        return refine_flow
