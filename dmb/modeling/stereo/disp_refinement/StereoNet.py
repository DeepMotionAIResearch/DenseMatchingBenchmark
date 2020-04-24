import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.edge_aware import  EdgeAwareRefinement

class StereoNetRefinement(nn.Module):
    """
    The disparity refinement module proposed in StereoNet.

    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
        num (int): the number of edge aware refinement module

    Inputs:
        disps (list of Tensor): estimated disparity map, in [BatchSize, 1, Height//s, Width//s] layout
        left (Tensor): left image feature, in [BatchSize, Channels, Height, Width] layout
        right(Tensor): right image feature, in [BatchSize, Channels, Height, Width] layout
        leftImage (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        rightImage (Tensor): right image, in [BatchSize, 3, Height, Width] layout

    Outputs:
        refine_disps (list of Tensor): refined disparity map, in [BatchSize, 1, Height, Width] layout

    """

    def __init__(self, in_planes, batch_norm=True, num=1):
        super(StereoNetRefinement, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm
        self.num = num

        # cascade the edge aware refinement module
        self.refine_blocks = nn.ModuleList([
            EdgeAwareRefinement(self.in_planes, self.batch_norm) for _ in range(self.num)
        ])

    def forward(self, disps, left, right, leftImage, rightImage):
        # only one estimated disparity map in StereoNet
        init_disp = disps[-1]

        # Upsample the coarse disparity map to the full resolution
        h, w = leftImage.shape[-2:]

        # the scale of downsample
        scale = w / init_disp.shape[-1]

        # upsample disparity map to image size, in [BatchSize, 1, Height, Width]
        init_disp = F.interpolate(init_disp, size=(h, w), mode='bilinear', align_corners=False)
        init_disp = init_disp * scale

        # cascade and refine the previous disparity map
        refine_disps = [init_disp]
        for block in self.refine_blocks:
            refine_disps.append(block(refine_disps[-1], leftImage))

        # In this framework, we always keep the better disparity map be ahead the worse.
        refine_disps.reverse()

        return refine_disps




