import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import bn_relu_conv


class AnyNetBackbone(nn.Module):
    """
    Backbone proposed in AnyNet.
    Args:
        in_planes (int): the channels of input
        C (int): the base channels of convolution layer in AnyNet
        block_num (int): the number of blocks.
        batch_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms (list of Tensor): left image feature maps in layout
                                [BatchSize, 8C, Height//16, Width//16] and
                                [BatchSize, 4C, Height//8, Width//8] and
                                [BatchSize, 2C, Height//4, Width//4]

        r_fms (list of Tensor): right image feature maps in layout
                                [BatchSize, 8C, Height//16, Width//16] and
                                [BatchSize, 4C, Height//8, Width//8] and
                                [BatchSize, 2C, Height//4, Width//4]
    """

    def __init__(self, in_planes=3, C=1, block_num=2, batch_norm=True):
        super(AnyNetBackbone, self).__init__()
        self.in_planes = in_planes
        self.C = C
        self.block_num = block_num
        self.batch_norm = batch_norm

        # input image down-sample to 1/4 resolution
        self.conv_4x = nn.Sequential(
            nn.Conv2d(in_planes, C, 3, 1, 1, dilation=1, bias=False),
            bn_relu_conv(batch_norm, C, C, 3, 2, 1, dilation=1, bias=False),
            self._make_down_blocks(batch_norm, C, 2*C, block_num),
        )

        # down-sample to 1/8 resolution
        self.conv_8x = self._make_down_blocks(batch_norm, in_planes=2*C,
                                              out_planes=4*C, block_num=block_num)
        # down-sample to 1/16 resolution
        self.conv_16x = self._make_down_blocks(batch_norm, in_planes=4*C,
                                               out_planes=8*C, block_num=block_num)
        self.conv_mix_8x = self._make_up_blocks(batch_norm, in_planes=12*C, out_planes=4*C)

        self.conv_mix_4x = self._make_up_blocks(batch_norm, in_planes=6*C, out_planes=2*C)


    def _make_down_blocks(self, batch_norm, in_planes, out_planes, block_num):
        blocks = []
        blocks.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        for i in range(block_num):
            blocks.append(bn_relu_conv(batch_norm, in_planes, out_planes,
                                      kernel_size=3, stride=1, padding=1, bias=False))
            in_planes = out_planes

        return nn.Sequential(*blocks)

    def _make_up_blocks(self, batch_norm, in_planes, out_planes):
        blocks = nn.Sequential(
            bn_relu_conv(batch_norm, in_planes, out_planes, kernel_size=3,
                         stride=1, padding=1, dilation=1, bias=False),
            bn_relu_conv(batch_norm, out_planes, out_planes, kernel_size=3,
                         stride=1, padding=1, dilation=1, bias=False),
        )
        return blocks

    def _forward(self, x):
        # in: [B, 3, H, W], out: [B, 2C, H/4, W/4]
        output_4x = self.conv_4x(x)

        # in: [B, 2C, H/4, W/4], out: [B, 4C, H/8, W/8]
        output_8x = self.conv_8x(output_4x)

        # in: [B, 4C, H/8, W/8], out: [B, 8C, H/16, W/16]
        output_16x = self.conv_16x(output_8x)

        h_8x, w_8x = output_8x.shape[-2:]
        # in: [B, 8C, H/16, W/16], out: [B, 8C, H/8, W/8]
        up_output_16x = F.interpolate(output_16x, size=(h_8x, w_8x), mode='bilinear', align_corners=False)
        # in: [B, 12C, H/8, W/8], out: [B, 4C, H/8, W/8]
        output_mix_8x = self.conv_mix_8x(torch.cat((output_8x, up_output_16x), dim=1))

        h_4x, w_4x = output_4x.shape[-2:]
        # in: [B, 4C, H/8, W/8], out: [B, 4C, H/4, W/4]
        up_output_mix_8x = F.interpolate(output_mix_8x, size=(h_4x, w_4x), mode='bilinear', align_corners=False)
        # in: [B, 6C, H/4, W/4], out: [B, 2C, H/4, W/4]
        output_mix_4x = self.conv_mix_4x(torch.cat((output_4x, up_output_mix_8x), dim=1))

        return [output_16x, output_mix_8x, output_mix_4x]

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input

        l_fms = self._forward(l_img)
        r_fms = self._forward(r_img)

        return l_fms, r_fms
