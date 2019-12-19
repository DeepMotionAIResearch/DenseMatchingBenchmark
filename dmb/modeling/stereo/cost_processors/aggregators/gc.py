import torch
import torch.nn as nn

from dmb.modeling.stereo.layers.basic_layers import conv3d_bn_relu, deconv3d_bn_relu


class GCAggregator(nn.Module):
    """
    Args:
        max_disp (int): max disparity
        in_planes (int): the channels of raw cost volume
        batch_norm (bool): whether use batch normalization layer, default is True
    Inputs:
        raw_cost (Tensor): concatenation-based cost volume without further processing,
            in [BatchSize, in_planes, max_disp//2, Height, Width] layout
    Outputs:
        cost_volume (list of Tensor): cost volume
            in [BatchSize, max_disp, Height, Width] layout
    """

    def __init__(self, max_disp, in_planes=64, batch_norm=True):
        super(GCAggregator, self).__init__()
        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm
        self.F = self.in_planes // 2

        self.layer19 = self._make_layer(self.in_planes, self.F)
        self.layer20 = self._make_layer(self.F, self.F)
        self.layer21 = self._make_layer(self.in_planes + self.F, self.F * 2, 2)
        self.layer22 = self._make_layer(self.F * 2, self.F * 2)
        self.layer23 = self._make_layer(self.F * 2, self.F * 2)
        self.layer24 = self._make_layer(self.F * 2 + self.F * 2, self.F * 2, 2)
        self.layer25 = self._make_layer(self.F * 2, self.F * 2)
        self.layer26 = self._make_layer(self.F * 2, self.F * 2)
        self.layer27 = self._make_layer(self.F * 2 + self.F * 2, self.F * 2, 2)
        self.layer28 = self._make_layer(self.F * 2, self.F * 2)
        self.layer29 = self._make_layer(self.F * 2, self.F * 2)
        self.layer30 = self._make_layer(self.F * 2 + self.F * 2, self.F * 4, 2)

        self.layer31 = self._make_layer(self.F * 4, self.F * 4)
        self.layer32 = self._make_layer(self.F * 4, self.F * 4)

        self.layer33 = self._make_tlayer(self.F * 4, self.F * 2)
        self.layer34 = self._make_tlayer(self.F * 2, self.F * 2)
        self.layer35 = self._make_tlayer(self.F * 2, self.F * 2)
        self.layer36 = self._make_tlayer(self.F * 2, self.F)
        self.layer37 = self._make_tlayer(self.F, 1, has_bn_relu=False)

    def _make_layer(self, in_planes, out_planes, stride=1):
        return conv3d_bn_relu(
            self.batch_norm, in_planes, out_planes,
            kernel_size=3, stride=stride, padding=1,
            dilation=1, bias=False
        )

    def _make_tlayer(self, in_planes, out_planes, stride=2, has_bn_relu=True):
        if has_bn_relu:
            return deconv3d_bn_relu(
                self.batch_norm, in_planes, out_planes,
                kernel_size=3, stride=stride, padding=1,
                output_padding=1, bias=False
            )
        else:
            return nn.ConvTranspose3d(
                in_planes, out_planes,
                kernel_size=3, stride=stride,
                padding=1, output_padding=1
            )

    def forward(self, raw_cost):
        # (BatchSize, Channels*2, max_disp/2, Height/2, Width/2), Channels = in_planes//2
        cost_volume18 = raw_cost
        # (BatchSize, Channels, max_disp/2, Height/2, Width/2)
        cost_volume19 = self.layer19(cost_volume18)
        # (BatchSize, Channels, max_disp/2, Height/2, Width/2)
        cost_volume20 = self.layer20(cost_volume19)
        # (BatchSize, Channels*2, max_disp/4, Height/4, Width/4)
        cost_volume21 = self.layer21(torch.cat([cost_volume18, cost_volume20], dim=1))

        # (BatchSize, Channels*2, max_disp/4, Height/4, Width/4)
        cost_volume22 = self.layer22(cost_volume21)
        # (BatchSize, Channels*2, max_disp/4, Height/4, Width/4)
        cost_volume23 = self.layer23(cost_volume22)
        # (BatchSize, Channels*2, max_disp/8, Height/8, Width/8)
        cost_volume24 = self.layer24(torch.cat([cost_volume21, cost_volume23], dim=1))

        # (BatchSize, Channels*2, max_disp/8, Height/8, Width/8)
        cost_volume25 = self.layer25(cost_volume24)
        # (BatchSize, Channels*2, max_disp/8, Height/8, Width/8)
        cost_volume26 = self.layer26(cost_volume25)
        # (BatchSize, Channels*2, max_disp/16, Height/16, Width/16)
        cost_volume27 = self.layer27(torch.cat([cost_volume24, cost_volume26], dim=1))

        # (BatchSize, Channels*2, max_disp/16, Height/16, Width/16)
        cost_volume28 = self.layer28(cost_volume27)
        # (BatchSize, Channels*2, max_disp/16, Height/16, Width/16)
        cost_volume29 = self.layer29(cost_volume28)
        # (BatchSize, Channels*4, max_disp/32, Height/32, Width/32)
        cost_volume30 = self.layer30(torch.cat([cost_volume27, cost_volume29], dim=1))

        # (BatchSize, Channels*4, max_disp/32, Height/32, Width/32)
        cost_volume31 = self.layer31(cost_volume30)
        # (BatchSize, Channels*4, max_disp/32, Height/32, Width/32)
        cost_volume32 = self.layer32(cost_volume31)

        # (BatchSize, Channels*2, max_disp/16, Height/16, Width/16)
        cost_volume33 = self.layer33(cost_volume32)
        # (BatchSize, Channels*2, max_disp/8, Height/8, Width/8)
        cost_volume34 = self.layer34(cost_volume33 + cost_volume29)
        # (BatchSize, Channels*2, max_disp/4, Height/4, Width/4)
        cost_volume35 = self.layer35(cost_volume34 + cost_volume26)
        # (BatchSize, Channels, max_disp/2, Height/2, Width/2)
        cost_volume36 = self.layer36(cost_volume35 + cost_volume23)
        # (BatchSize, 1, max_disp, Height, Width)
        cost_volume37 = self.layer37(cost_volume36 + cost_volume20)
        # (BatchSize, max_disp, Height, Width)
        cost_volume = cost_volume37.squeeze(dim=1)

        return [cost_volume]
