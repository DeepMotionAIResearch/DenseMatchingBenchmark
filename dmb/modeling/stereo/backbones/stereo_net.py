import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn, conv_bn_relu, BasicBlock

class DownsampleHead(nn.Module):
    """
    Args:
        in_planes (int): the channels of input
        out_planes (int): the channels of output
        batchNorm, (bool): whether use batch normalization layer, default True
    Inputs:
        x, (tensor): feature in (BatchSize, in_planes, Height, Width) layout
    Outputs:
        down_x, (tensor): downsampled feature in (BatchSize, out_planes, Height, Width) layout
    """

    def __init__(self, in_planes, out_planes, batch_norm=True):
        super(DownsampleHead, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.batch_norm = batch_norm

        self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=5,
                                    stride=2, padding=2, bias=True)

    def forward(self, x):
        down_x = self.downsample(x)
        return down_x


class StereoNetBackbone(nn.Module):
    """
    Backbone proposed in StereoNet.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
        downsample_num (int): the number of downsample module,
            the input RGB image will be downsample to 1/2^num resolution, default 3, i.e., 1/8 resolution
        residual_num (int): the number of residual blocks, used for robust feature extraction
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms (Tensor): left image feature maps, in [BatchSize, 32, Height//8, Width//8] layout
        r_fms (Tensor): right image feature maps, in [BatchSize, 32, Height//8, Width//8] layout
    """

    def __init__(self, in_planes=3, batch_norm=True, downsample_num=3, residual_num=6):
        super(StereoNetBackbone, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm
        self.downsample_num = downsample_num
        self.residual_num = residual_num

        # Continuously downsample the input RGB image to 1/2^num resolution
        in_planes = self.in_planes
        out_planes = 32

        self.downsample = nn.ModuleList()
        for _ in range(self.downsample_num):
            self.downsample.append(DownsampleHead(in_planes, out_planes))
            in_planes = out_planes
            out_planes = 32

        # Build residual feature extraction module
        self.residual_blocks = nn.ModuleList()
        for _ in range(self.residual_num):
            self.residual_blocks.append(BasicBlock(
                self.batch_norm, 32, 32, stride=1, downsample=None, padding=1, dilation=1
            ))

        self.lastconv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)


    def _forward(self, x):

        for i in range(self.downsample_num):
            x = self.downsample[i](x)

        for i in range(self.residual_num):
            x = self.residual_blocks[i](x)

        output_feature = self.lastconv(x)

        return output_feature

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input

        l_fms = self._forward(l_img)
        r_fms = self._forward(r_img)

        return l_fms, r_fms
