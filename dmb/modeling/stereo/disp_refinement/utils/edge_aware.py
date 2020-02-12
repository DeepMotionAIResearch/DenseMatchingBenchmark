import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn, conv_bn_relu, BasicBlock


class EdgeAwareRefinement(nn.Module):
    """
    The edge aware refinement module proposed in StereoNet.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        disp (Tensor): estimated disparity map, in [BatchSize, 1, Height//s, Width//s] layout
        leftImage (Tensor): left image, in [BatchSize, Channels, Height, Width] layout

    Outputs:
        refine_disp (Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout
    """

    def __init__(self, in_planes, batch_norm=True):
        super(EdgeAwareRefinement, self).__init__()

        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv_mix = conv_bn_relu(self.batch_norm, self.in_planes, 32,
                                      kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        # Dilated residual module
        self.residual_dilation_blocks = nn.ModuleList()
        self.dilation_list = [1, 2, 4, 8, 1, 1]
        for dilation in self.dilation_list:
            self.residual_dilation_blocks.append(
                BasicBlock(self.batch_norm, 32, 32, stride=1, downsample=None,
                           padding=1, dilation=dilation)
            )

        self.conv_res = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, disp, leftImage):
        h, w = leftImage.shape[-2:]

        # the scale of downsample
        scale = w / disp.shape[-1]

        # upsample disparity map to image size, in [BatchSize, 1, Height, Width]
        up_disp = F.interpolate(disp, size=(h, w), mode='bilinear', align_corners=False)
        up_disp = up_disp * scale

        # residual refinement
        # mix the info inside the disparity map and left image
        mix_feat = self.conv_mix(torch.cat((up_disp, leftImage), dim=1))

        for block in self.residual_dilation_blocks:
            mix_feat = block(mix_feat)

        # get residual disparity map, in [BatchSize, 1, Height, Width]
        res_disp = self.conv_res(mix_feat)

        # refine the upsampled disparity map, in [BatchSize, 1, Height, Width]
        refine_disp = res_disp + up_disp

        # promise all disparity value larger than 0, in [BatchSize, 1, Height, Width]
        refine_disp = F.relu(refine_disp, inplace=True)

        return refine_disp


