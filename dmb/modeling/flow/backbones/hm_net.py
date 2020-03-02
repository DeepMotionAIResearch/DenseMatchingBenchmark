import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu

class backboneHead(nn.Module):
    def __init__(self, in_planes, out_planes, batch_norm=True):
        super(backboneHead, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.batch_norm = batch_norm

        self.feature_extraction_level = nn.Sequential(
            conv_bn_relu(batch_norm, in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bias=True),
            conv_bn_relu(batch_norm, out_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            conv_bn_relu(batch_norm, out_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
        )

    def forward(self, x):
        # in: [B, in_planes, H, W], out: [B, out_planes, H//2, W//2]
        x = self.feature_extraction_level(x)
        return x


class hybridHead(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(hybridHead, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm =batch_norm
        self.hybrid_feature_extraction_level = nn.Sequential(
            conv_bn_relu(batch_norm, in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=1, bias=True),
            conv_bn_relu(batch_norm, out_planes, out_planes, kernel_size=kernel_size, stride=1,
                         padding=padding, dilation=1, bias=True),
        )

    def forward(self, x):
        # in: [B, in_planes, H, W], out: [B, out_planes, H//2, W//2]
        x = self.hybrid_feature_extraction_level(x)
        return x


class HMNetBackbone(nn.Module):
    """
    Backbone proposed in HMNet.
    Args:
        in_planes (int): the channels of input
        batch_norm (bool): whether use batch normalization layer, default True
    Inputs:
        l_img (Tensor): left image, in [BatchSize, 3, Height, Width] layout
        r_img (Tensor): right image, in [BatchSize, 3, Height, Width] layout
    Outputs:
        l_fms (Tensor): left image feature maps, in
                        [BatchSize,  32, Height//4, Width//4],
                        [BatchSize,  64, Height//8, Width//8],
                        [BatchSize,  96, Height//16, Width//16],
                        [BatchSize, 128, Height//32, Width//32],
                        [BatchSize, 196, Height//64, Width//64] layout

        r_fms (Tensor): right image feature maps, in
                        [BatchSize,  32, Height//4, Width//4],
                        [BatchSize,  64, Height//8, Width//8],
                        [BatchSize,  96, Height//16, Width//16],
                        [BatchSize, 128, Height//32, Width//32],
                        [BatchSize, 196, Height//64, Width//64] layout
        hybrid_fms (Tensor): hybrid image feature maps, in
                        [BatchSize,  32, Height//4, Width//4],
                        [BatchSize,  64, Height//8, Width//8],
                        [BatchSize,  96, Height//16, Width//16],
                        [BatchSize, 128, Height//32, Width//32],
                        [BatchSize, 196, Height//64, Width//64] layout
    """

    def __init__(self, in_planes=3, batch_norm=True):
        super(HMNetBackbone, self).__init__()
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.feature_planes_list = [in_planes, 16, 32, 64, 96, 128, 196]
        self.feature_extractions = nn.ModuleList()
        for lvl in range(1, len(self.feature_planes_list)):
             self.feature_extractions.append(backboneHead(in_planes=self.feature_planes_list[lvl-1],
                                                          out_planes=self.feature_planes_list[lvl],
                                                          batch_norm=batch_norm))

        self.hybrid_feature_planes_list = [2*in_planes, 16, 32, 64, 96, 128, 196]
        self.hybrid_kernel_list = [None, 7, 5, 5, 3, 3, 3]
        self.hybrid_feature_extractions = nn.ModuleList()
        for lvl in range(1, len(self.feature_planes_list)):
            self.hybrid_feature_extractions.append(hybridHead(in_planes=self.hybrid_feature_planes_list[lvl-1],
                                                              out_planes=self.hybrid_feature_planes_list[lvl],
                                                              kernel_size=self.hybrid_kernel_list[lvl],
                                                              stride=2,
                                                              padding=self.hybrid_kernel_list[lvl]//2,
                                                              batch_norm=batch_norm))

    def _forward(self, x):

        x_2  = self.feature_extractions[0](x)
        x_4  = self.feature_extractions[1](x_2)
        x_8  = self.feature_extractions[2](x_4)
        x_16 = self.feature_extractions[3](x_8)
        x_32 = self.feature_extractions[4](x_16)
        x_64 = self.feature_extractions[5](x_32)

        return [x_4, x_8, x_16, x_32, x_64]

    def _hybrid_forward(self, x):
        x_2  = self.hybrid_feature_extractions[0](x)
        x_4  = self.hybrid_feature_extractions[1](x_2)
        x_8  = self.hybrid_feature_extractions[2](x_4)
        x_16 = self.hybrid_feature_extractions[3](x_8)
        x_32 = self.hybrid_feature_extractions[4](x_16)
        x_64 = self.hybrid_feature_extractions[5](x_32)

        return [x_4, x_8, x_16, x_32, x_64]

    def forward(self, *input):
        if len(input) != 2:
            raise ValueError('expected input length 2 (got {} length input)'.format(len(input)))

        l_img, r_img = input

        l_fms = self._forward(l_img)
        r_fms = self._forward(r_img)

        hybrid_fms = self._hybrid_forward(torch.cat((l_img, r_img), dim=1))

        return l_fms, r_fms, hybrid_fms
