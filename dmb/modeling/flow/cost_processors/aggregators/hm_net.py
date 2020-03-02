import torch
import torch.nn as nn
import numpy as np

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu

class HMAggregator(nn.Module):
    """
    Args:
        in_planes, (int): in channels of input
        out_planes, (int): out channels of output
        agg_planes_list, (list): the channels of intermediate convolution layers
        dense, (bool): whether use dense connection
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        cost, (tensor): cost volume in [BatchSize, in_planes, Height, Width] layout

    Outputs:
        cost, (tensor): aggregated cost volume in [BatchSize, out_planes, Height, Width] layout
    """
    def __init__(self, in_planes, out_planes, agg_planes_list, dense, batch_norm=True):
        super(HMAggregator, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.dense = dense
        self.batch_norm =batch_norm

        self.agg_planes_list = agg_planes_list
        self.in_planes_list = [in_planes] + self.agg_planes_list

        if dense:
            self.in_planes_list = np.cumsum(self.in_planes_list).tolist()

        self.agg_list = nn.ModuleList()
        for i in range(len(self.agg_planes_list)):
            self.agg_list.append(conv_bn_relu(batch_norm, self.in_planes_list[i], self.agg_planes_list[i],
                                              kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.agg_final = conv_bn_relu(batch_norm, self.in_planes_list[-1], self.out_planes,
                                          kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def forward(self, cost):
        for agg in self.agg_list:
            out = agg(cost)
            if self.dense:
                cost = torch.cat((cost, out), dim=1)
            else:
                cost = out
        # [B, out_planes, H, W]
        cost = self.agg_final(cost)

        return cost

