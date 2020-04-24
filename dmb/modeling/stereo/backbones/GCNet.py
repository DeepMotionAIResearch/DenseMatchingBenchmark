import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu, BasicBlock


class GCNetBackbone(nn.Module):
    """
    Backbone proposed in GCNet.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width]
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width]
    Outputs:
        l_fms (Tensor): left image feature maps, in [BatchSize, 32, Height//2, Width//2]
        right (Tensor): right image feature maps, in [BatchSize, 32, Height//2, Width//2]
    """

    def __init__(self, in_planes, batch_norm=True):
        super(GCNetBackbone, self).__init__()
        self.in_planes = in_planes

        self.backbone = nn.Sequential(
            conv_bn_relu(batch_norm, self.in_planes, 32, 5, 2, 2),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            BasicBlock(batch_norm, 32, 32, 1, None, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1)
        )

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))
        l_img, r_img = input

        l_fms = self.backbone(l_img)
        r_fms = self.backbone(r_img)

        return l_fms, r_fms
