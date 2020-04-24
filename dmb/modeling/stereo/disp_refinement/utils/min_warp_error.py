"""
Written by youmi.
Implementation of stack dilation module.

FrameWork: PyTorch
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from dmb.modeling.stereo.layers.inverse_warp import inverse_warp
from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu, BasicBlock, conv_bn, deconv_bn_relu


class WarpErrorRefinement(nn.Module):
    """
    Minimise the warp error to refine initial disparity map.
    Args:
        in_planes, (int): the channels of left feature
        batch_norm, (bool): whether use batch normalization layer

    Inputs:
        disp, (Tensor): the left disparity map, in (BatchSize, 1, Height//s, Width//s) layout
        left, (Tensor): the left image feature, in (BatchSize, Channels, Height, Width) layout
        right, (Tensor): the right image feature, in (BatchSize, Channels, Height, Width) layout

    Outputs:
        refine_disp (Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout

    """

    def __init__(self, in_planes, C=16, batch_norm=True):
        super(WarpErrorRefinement, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm
        self.C = C

        self.conv_mix = conv_bn_relu(batch_norm, in_planes*4 + 1, 2*C, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        # Dilated residual module
        self.residual_dilation_blocks = nn.ModuleList()
        self.dilation_list = [1, 2, 4, 8, 1, 1]
        for dilation in self.dilation_list:
            self.residual_dilation_blocks.append(
                conv_bn_relu(batch_norm, 2*C, 2*C, kernel_size=3, stride=1,
                             padding=dilation, dilation=dilation, bias=False)
            )

        self.conv_res = nn.Conv2d(2*C, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, disp, left, right):
        B, C, H, W = left.shape

        # the scale of downsample
        scale = W / disp.shape[-1]

        # upsample disparity map to image size, in [BatchSize, 1, Height, Width]
        up_disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=True)
        up_disp = up_disp * scale

        # calculate warp error
        warp_right = inverse_warp(right, -up_disp)
        error = torch.abs(left - warp_right)

        # residual refinement
        # mix the info inside the disparity map, left image, right image and warp error
        mix_feat = self.conv_mix(torch.cat((left, right, warp_right, error, disp), 1))

        for block in self.residual_dilation_blocks:
            mix_feat = block(mix_feat)

            # get residual disparity map, in [BatchSize, 1, Height, Width]
        res_disp = self.conv_res(mix_feat)

        # refine the upsampled disparity map, in [BatchSize, 1, Height, Width]
        refine_disp = res_disp + up_disp

        # promise all disparity value larger than 0, in [BatchSize, 1, Height, Width]
        refine_disp = F.relu(refine_disp, inplace=True)

        return refine_disp
