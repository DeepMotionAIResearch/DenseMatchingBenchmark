import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu
from dmb.ops import GateRecurrent2dnoind

class AnyNetRefinement(nn.Module):
    """

    The disparity refinement module proposed in AnyNet.

    Args:
        in_planes (int): the channels of input
        spn_planes (int): the channels used for spn
        batch_norm (bool): whether use batch normalization layer, default True

    Inputs:
        disps (list of Tensor): estimated disparity map, in [BatchSize, 1, Height//s, Width//s] layout
        left (Tensor): left image feature, in [BatchSize, Channels, Height, Width] layout
        right(Tensor): right image feature, in [BatchSize, Channels, Height, Width] layout
        leftImage (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        rightImage (Tensor): right image, in [BatchSize, 3, Height, Width] layout

    Outputs:
        refine_disps (list of Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout


    """
    def __init__(self, in_planes, spn_planes=8, batch_norm=True):
        super(AnyNetRefinement, self).__init__()
        self.in_planes = in_planes
        self.spn_planes = spn_planes
        self.batch_norm = batch_norm

        self.img_conv = nn.Sequential(
            conv_bn_relu(batch_norm, in_planes, spn_planes * 2,
                         kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            conv_bn_relu(batch_norm, spn_planes * 2, spn_planes * 2,
                         kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            conv_bn_relu(batch_norm, spn_planes * 2, spn_planes * 2,
                         kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.Conv2d(spn_planes * 2, spn_planes * 3,
                         kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        )

        self.disp_conv = nn.Conv2d(1, spn_planes, kernel_size=3,
                                   stride=1, padding=1, dilation=1, bias=False)

        self.classify = nn.Conv2d(spn_planes, 1, kernel_size=3,
                                   stride=1, padding=1, dilation=1, bias=False)

        # left->right propagation
        self.spn = GateRecurrent2dnoind(True,False)


    def forward(self, disps, left, right, leftImage, rightImage):
        # only disparity map in the last stage need to be refined
        init_disp = disps[-1]

        # down-sample the left image to the resolution of disparity map
        h, w = init_disp.shape[-2:]
        leftImage = F.interpolate(leftImage, size=(h,w), mode='bilinear', align_corners=False)

        # extract guidance information from left image
        # [B, spn_planes*3, H, W]
        G = self.img_conv(leftImage)

        # G1~G3: three coefficient maps (e.g., left-top, left-center, left-bottom)
        # [B, spn_planes, H, W]
        G1, G2, G3 = torch.split(G, self.spn_planes, dim=1)

        # for any pixel i, |G1(i)| + |G2(i)| + |G3(i)| <= 1 is a sufficient condition for model stability
        # [B, spn_planes, H, W]
        sum_abs = G1.abs() + G2.abs() + G3.abs()
        G1 = torch.div(G1, sum_abs + 1e-8)
        G2 = torch.div(G2, sum_abs + 1e-8)
        G3 = torch.div(G3, sum_abs + 1e-8)

        # [B, spn_planes, H, W]
        disp_feat = self.disp_conv(init_disp)

        # [B, spn_planes, H, W]
        propogated_disp_feat = self.spn(disp_feat, G1, G2, G3)

        # [B, 1, H, W]
        res_disp = self.classify(propogated_disp_feat)

        # [B, 1, H, W]
        refine_disp = F.relu(res_disp + init_disp)

        disps.append(refine_disp)
        # In this framework, we always keep the better disparity map be ahead the worse.
        disps.reverse()

        return disps



