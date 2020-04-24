import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, in_planes, mid_planes, out_planes, dilation_rate, dropout_rate, bn_start=True, batch_norm=True):
        super(_DenseAsppBlock, self).__init__()
        self.bn_start = bn_start
        self.batch_norm = batch_norm
        if bn_start:
            self.bn0 = nn.BatchNorm2d(in_planes, momentum=0.0003)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(mid_planes, momentum=0.0003)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3,
                               stride=1, padding=dilation_rate, dilation=dilation_rate)

        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.bn_start:
            x = self.bn0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x

class DenseAspp(nn.Module):

    def __init__(self, in_planes, out_planes, dropout_rate=0., batch_norm=True):
        super(DenseAspp, self).__init__()
        self.batch_norm = batch_norm
        in_planes = int(in_planes)
        in_2_planes = int(in_planes / 2)
        in_4_planes = int(in_planes / 4)

        self.ASPP_3  = _DenseAsppBlock(in_planes=in_planes, mid_planes=in_2_planes, out_planes=in_4_planes,
                                      dilation_rate=3,  dropout_rate=dropout_rate, bn_start=False, batch_norm=batch_norm)

        self.ASPP_6  = _DenseAsppBlock(in_planes=in_planes + in_4_planes * 1 , mid_planes=in_2_planes,
                                      out_planes=in_4_planes,
                                      dilation_rate=6,  dropout_rate=dropout_rate, bn_start=batch_norm, batch_norm=batch_norm)

        self.ASPP_12 = _DenseAsppBlock(in_planes=in_planes + in_4_planes * 2, mid_planes=in_2_planes,
                                      out_planes=in_4_planes,
                                      dilation_rate=12, dropout_rate=dropout_rate, bn_start=batch_norm, batch_norm=batch_norm)

        self.ASPP_18 = _DenseAsppBlock(in_planes=in_planes + in_4_planes * 3, mid_planes=in_2_planes,
                                      out_planes=in_4_planes,
                                      dilation_rate=18, dropout_rate=dropout_rate, bn_start=batch_norm, batch_norm=batch_norm)

        self.ASPP_24 = _DenseAsppBlock(in_planes=in_planes + in_4_planes * 4, mid_planes=in_2_planes,
                                      out_planes=in_4_planes,
                                      dilation_rate=24, dropout_rate=dropout_rate, bn_start=batch_norm, batch_norm=batch_norm)

        self.fuse_conv = nn.Sequential(conv_bn_relu(batch_norm, in_planes + in_4_planes * 5, in_planes,
                                                   kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                      nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, dilation=1, bias=False))

    def forward(self, feature):
        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)
        feature = torch.cat((aspp24, feature), dim=1)

        feature = self.fuse_conv(feature)

        return feature