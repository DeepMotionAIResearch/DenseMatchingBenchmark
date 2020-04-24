import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn, conv_bn_relu


class RefinementHeand(nn.Module):
    """
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        disps (Tensor): estimated disparity map, in [BatchSize, 1, Height, Width] layout
        input (Tensor): feature used to guide refinement, in [BatchSize, in_planes, Height, Width] layout

    Outputs:
        refine_disp (Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout

    """
    def __init__(self, in_planes, batch_norm=True):
        super(RefinementHeand, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv = nn.Sequential(
            conv_bn_relu(batch_norm, in_planes, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            conv_bn_relu(batch_norm, 32, 16, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            conv_bn_relu(batch_norm, 16, 16, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            conv_bn_relu(batch_norm, 16, 16, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.classify = nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, init_disp, input):

        res_disp = self.classify(self.conv(input))

        refine_disp = F.relu(res_disp + init_disp)

        return refine_disp


class DeepPrunerRefinement(nn.Module):
    """
    The disparity refinement module proposed in DeepPruner.

    Args:
        in_planes (list, tuple): the channels of input of each refinement sub network
        batch_norm (bool): whether use batch normalization layer, default True
        num (int): the number of cascade refinement sub network, default 1

    Inputs:
        disps (list of Tensor): estimated disparity map, in [BatchSize, 1, Height, Width] layout
        input (Tensor): feature used to guide refinement, in [BatchSize, in_planes, Height, Width] layout

    Outputs:
        refine_disps (list of Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout

    """

    def __init__(self, in_planes_list, batch_norm=True, num=1):
        super(DeepPrunerRefinement, self).__init__()
        self.in_planes_list = in_planes_list
        self.batch_norm = batch_norm
        self.num = num

        # cascade the edge aware refinement module
        self.refine_blocks = nn.ModuleList([
            RefinementHeand(self.in_planes_list[i], self.batch_norm) for i in range(self.num)
        ])


    def forward(self, disps, low_ref_group_fms):

        for i in range(self.num):
            # get last stage disparity map
            init_disp = disps[-1]
            # concatenate last stage disparity map into guide feature map
            guide_fms = torch.cat((low_ref_group_fms[i], init_disp), dim=1)
            # residual refinement
            refine_disp = self.refine_blocks[i](init_disp, guide_fms)
            # up-sample the refined disparity map
            refine_disp = F.interpolate(refine_disp*2, scale_factor=(2,2), mode='bilinear', align_corners=False)

            disps.append(refine_disp)

        # In this framework, we always keep the better disparity map be ahead the worse.
        disps.reverse()

        return disps




