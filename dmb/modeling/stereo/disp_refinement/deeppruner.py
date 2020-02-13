import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn, conv_bn_relu


class DeepPrunerRefinement(nn.Module):
    """
    The disparity refinement module proposed in DeepPruner.

    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        disps (list of Tensor): estimated disparity map, in [BatchSize, 1, Height, Width] layout
        input (Tensor): feature used to guide refinement, in [BatchSize, in_planes, Height, Width] layout

    Outputs:
        refine_disps (list of Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout

    """

    def __init__(self, in_planes, batch_norm=True):
        super(DeepPrunerRefinement, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv = nn.Sequential(
            conv_bn_relu(batch_norm, in_planes, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            conv_bn_relu(batch_norm, 32, 16, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            conv_bn_relu(batch_norm, 16, 16, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            conv_bn_relu(batch_norm, 16, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=False))

        self.classify = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, disps, input):
        # only one estimated disparity map in StereoNet
        init_disp = disps[-1]

        res_disp = self.classify(self.conv(init_disp))

        refine_disp = F.relu(res_disp + init_disp)

        refine_disps = [refine_disp]

        return refine_disps




