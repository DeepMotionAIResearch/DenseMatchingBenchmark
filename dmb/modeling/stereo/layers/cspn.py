"""
Written by youmi
Implementation of CSPN.
Reference:
    CSPN: https://github.com/XinJCheng/CSPN/blob/master/cspn.py
    SPN: https://github.com/Liusifei/pytorch_spn/blob/master/left_right_demo.py
Time:
    For 2D:
        As paper report, 4 iterations of CSPN on one 1024*768 image only takes 3.689ms, 1514MiB memory if requires_grad=True
        But for 24 iterations, it takes 175.1ms, 5953MiB memory if requires_grad=True
        For our implementation, it takes 2.288ms, 763MiB memory if requires_grad=True
        But for 24 iterations of our implementation, it takes 13.109ms, 843MiB memory if requires_grad=True
        We also experiment SPP of PSMNet with the [BatchSize, 32, 256//4, 512//4] layout, BatchSize=1
        For a AvgPooling2d(kernerl_size=64, stride=64),
        1 iteration takes 151.861ms, 1433 memory if requires_grad=True
    For 3D:
        It mainly used to refine SPP module in PSMNet.
        SPP in PSMNet with the [BatchSize, 32, 256//4, 512//4] layout, after extending 4 pooling feature to 5D feature map,
            the layout becomes [BatchSize, 32, 4, 256//4, 512//4]. BatchSize=1
        In our implementation and Time consumption testing,
            input feature in [1, 32, 4, 256//4, 512//4] layout,
            affinity in [1, 27, 4, 256//4, 512//4] layout,
            24 iterations take 49.131ms, 793MiB memory if requires_grad=True
        However, if we just concatenate 4 pooling feature,
            the layout becomes [BatchSize, 32*4, 256//4, 512//4].
        In our implementation and Time consumption testing,
            input feature in [1, 32*4, 256//4, 512//4] layout,
            affinity in [1, 9, 256//4, 512//4] layout,
            24 iterations take 15.377ms, 709MiB memory if requires_grad=True
FrameWork: PyTorch
"""
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _triple


class AffinityPropagate(nn.Module):
    """
    Args:
        op (string): Operation name, optional in ['Conv2d', 'Conv3d']. Default: 'Conv2d'
        iterations (int):  Iteration times of propagation. Default: 1
        kernel_size (int or tuple): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 1
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        affinity: Tensor, in [BatchSize, Channels, Height, Width] layout, the affinity matrix
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout, any kind of feature map, e.g. depth map, segmentation
    Attention:
        1. affinity's Channels should equal with kerner_size*kerner_size for 2D
           affinity's Channels should equal with kerner_size*kerner_size*kernerl_size for 3D
        2. dilation should be larger than 0
        3. padding is not used in this version of implementation, so, random set is allowed...
    Outputs:
        feature: Tensor, in [BatchSize, Channels, Height, Width] layout,  feature after refined with affinity propagation
    """

    def __init__(self, op='Conv2d', iterations=1, kernel_size=3, stride=1, padding=1, dilation=1):
        super(AffinityPropagate, self).__init__()
        self.op = op
        self.iterations = iterations
        self.get_repeat_operation()
        self.kernel_size = self.repeat_op(kernel_size)
        self.stride = self.repeat_op(stride)
        self.padding = self.repeat_op(padding)
        self.dilation = self.repeat_op(dilation)
        self.get_pad_operation()
        self.get_pooling_operation()

    def forward(self, affinity, feature):
        # checkout wether affinity and feature satisfy our requiresments
        self.checkout(affinity, feature)

        # normalize affinity matrix
        affinity_abs = affinity.abs()
        affinity_sum = affinity_abs.sum(dim=1, keepdim=True)
        affinity_norm = torch.div(affinity_abs, affinity_sum)

        for it in range(self.iterations):

            # through padding, we can move to correspond direction by index
            feature_pad = self.pad_op(feature)
            # index the affinity matrix
            index = 0
            if self.op in ['Conv2d']:
                h, w = feature.shape[2:]
                for k_h in range(self.kernel_size[0]):
                    for k_w in range(self.kernel_size[1]):
                        st_h = k_h * self.dilation[0]
                        ed_h = st_h + h
                        st_w = k_w * self.dilation[1]
                        ed_w = st_w + w

                        if index == 0:
                            feature = feature_pad[:, :, st_h:ed_h, st_w:ed_w] \
                                      * affinity_norm[:, index:index + 1]
                        else:
                            feature += feature_pad[:, :, st_h:ed_h, st_w:ed_w] \
                                       * affinity_norm[:, index:index + 1]
                        index += 1

            if self.op in ['Conv3d']:
                d, h, w = feature.shape[2:]
                for k_d in range(self.kernel_size[0]):
                    for k_h in range(self.kernel_size[1]):
                        for k_w in range(self.kernel_size[2]):
                            st_d = k_d * self.dilation[0]
                            ed_d = st_d + d
                            st_h = k_h * self.dilation[1]
                            ed_h = st_h + h
                            st_w = k_w * self.dilation[2]
                            ed_w = st_w + w
                            if index == 0:
                                feature = feature_pad[:, :, st_d:ed_d, st_h:ed_h, st_w:ed_w] \
                                          * affinity_norm[:, index:index + 1]
                            else:
                                feature += feature_pad[:, :, st_d:ed_d, st_h:ed_h, st_w:ed_w] \
                                           * affinity_norm[:, index:index + 1]
                            index += 1

        if self.pooling_op is not None:
            feature = self.pooling_op(feature)

        return feature

    def get_repeat_operation(self):
        if self.op in ['Conv2d']:
            self.repeat_op = _pair
        if self.op in ['Conv3d']:
            self.repeat_op = _triple

    def get_pad_operation(self):
        if self.op in ['Conv2d']:
            lr = (self.dilation[1]) * (self.kernel_size[1] // 2)
            hw = (self.dilation[0]) * (self.kernel_size[0] // 2)
            self.pad_op = nn.ConstantPad2d((lr, lr, hw, hw), 0)
        if self.op in ['Conv3d']:
            lr = (self.dilation[2]) * (self.kernel_size[2] // 2)
            hw = (self.dilation[1]) * (self.kernel_size[1] // 2)
            fb = (self.dilation[0]) * (self.kernel_size[0] // 2)  # (front, back) => depth dimension
            self.pad_op = nn.ConstantPad3d((lr, lr, hw, hw, fb, fb), 0)

    def get_pooling_operation(self):
        self.pooling_op = None
        if self.op in ['Conv2d']:
            if self.stride[0] > 1 or self.stride[1] > 1:
                self.pooling_op = nn.AvgPool2d(self.kernel_size, self.stride)
        if self.op in ['Conv3d']:
            if self.stride[0] > 1 or self.stride[1] > 1 or self.stride[2] > 1:
                self.pooling_op = nn.AvgPool3d(self.kernel_size, self.stride)

    def checkout(self, affinity, feature):
        assert affinity.dim() == feature.dim(), \
            'affinity matrix should have same number of dimension as feature, ' \
            'but got affinity.dim()={} and feature.dim()={}' \
                .format(affinity.dim(), feature.dim())
        assert affinity.shape[0] == feature.shape[0], \
            'affinity matrix BatchSize should be same as feature, ' \
            'but got affinity BatchSize={} and feature BatchSize={}' \
                .format(affinity.shape[0], feature.shape[0])

        if self.op in ['Conv2d']:
            channels = self.kernel_size[0] * self.kernel_size[1]
            assert affinity.shape[1] == channels, \
                'affnity matrix should have {} channels, ' \
                'but got {}'.format(channels, affinity.shape[1])
            assert affinity.shape[2:] == feature.shape[2:], \
                'affinity matrix Height and Width should be same as feature, ' \
                'but got affinity H,W={},{} and feature H,W={},{}' \
                    .format(affinity.shape[2], affinity.shape[3], feature.shape[2], feature.shape[3])
            assert self.dilation[0] > 0 and self.dilation[1] > 0, 'dilation should be larger than 0'

        if self.op in ['Conv3d']:
            channels = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            assert affinity.shape[1] == channels, \
                'affnity matrix should have {} channels, ' \
                'but got {}'.format(channels, affinity.shape[1])
            assert affinity.shape[2:] == feature.shape[2:], \
                'affinity matrix Depth, Height and Width should be same as feature, ' \
                'but got affinity D,H,W={},{},{} and feature D,H,W={},{},{}' \
                    .format(
                    affinity.shape[2], affinity.shape[3], affinity.shape[4],
                    feature.shape[2], feature.shape[3], feature.shape[4]
                )
            assert self.dilation[0] > 0 and self.dilation[1] > 0 and self.dilation[2] > 0, \
                'dilation should be larger than 0'
