import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu
from dmb.modeling.stereo.cost_processors.utils.correlation1d_cost import correlation1d_cost
from dmb.modeling.stereo.disp_predictors.faster_soft_argmin import FasterSoftArgmin
from dmb.modeling.stereo.cost_processors.utils.hourglass_2d import Hourglass2D
from dmb.ops import ModulatedDeformConv
from dmb.modeling.stereo.disp_samplers.deep_prunner import UniformSampler

class DeformOffsetNet(nn.Module):
    def __init__(self, in_planes, disparity_sample_number=3,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, C=16, batch_norm=True):
        super(DeformOffsetNet, self).__init__()

        self.in_planes = in_planes
        self.disparity_sample_number = disparity_sample_number
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.C = C

        self.conv = conv_bn_relu(batch_norm, in_planes, 2*C, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.min_disparity_dcn = nn.ModuleDict({
            'context': Hourglass2D(in_planes=2 * C, batch_norm=batch_norm),
            'concat_context': conv_bn_relu(batch_norm, 4 * C, 2 * C, kernel_size=3, stride=1, padding=1, dilation=1,
                                           bias=False),
            'offset': nn.Conv2d(2 * C, 3 * kernel_size * kernel_size, kernel_size=3,
                                stride=1, padding=1, dilation=1, bias=False),
            'dcn': ModulatedDeformConv(1, 1, kernel_size, stride, padding, dilation, bias=bias),

        })
        self.max_disparity_dcn = nn.ModuleDict({
            'context': Hourglass2D(in_planes=2 * C, batch_norm=batch_norm),
            'concat_context': conv_bn_relu(batch_norm, 4 * C, 2 * C, kernel_size=3, stride=1, padding=1, dilation=1,
                                           bias=False),
            'offset': nn.Conv2d(2 * C, 3 * kernel_size * kernel_size, kernel_size=3,
                                stride=1, padding=1, dilation=1, bias=False),
            'dcn': ModulatedDeformConv(1, 1, kernel_size, stride, padding, dilation, bias=bias),

        })

        self.uniform_sampler = UniformSampler(disparity_sample_number=disparity_sample_number-1)

    def forward(self, context, base_disparity, confidence_cost):
        context = self.conv(context)

        hourglass_context, _, _ = self.min_disparity_dcn['context'](context, presqu=None, postsqu=None)
        min_context = self.min_disparity_dcn['concat_context'](torch.cat((context, hourglass_context), dim=1))
        min_offset = self.min_disparity_dcn['offset'](min_context)
        oh, ow, mask = torch.chunk(min_offset, chunks=3, dim=1)
        min_mask = torch.sigmoid(mask)
        min_offset = torch.cat((oh, ow), dim=1)
        min_disparity = self.min_disparity_dcn['dcn'](base_disparity, min_offset, min_mask)

        hourglass_context, _, _ = self.max_disparity_dcn['context'](context, presqu=None, postsqu=None)
        max_context = self.max_disparity_dcn['concat_context'](torch.cat((context, hourglass_context), dim=1))
        max_offset = self.max_disparity_dcn['offset'](max_context)
        oh, ow, mask = torch.chunk(max_offset, chunks=3, dim=1)
        max_mask = torch.sigmoid(mask)
        max_offset = torch.cat((oh, ow), dim=1)
        max_disparity = self.max_disparity_dcn['dcn'](base_disparity, max_offset, max_mask)

        disparity_sample = self.uniform_sampler(min_disparity, max_disparity)
        disparity_sample = torch.cat((base_disparity, disparity_sample), dim=1)

        return disparity_sample, \
               [min_offset, max_offset], \
               [min_mask, max_mask], \
               [min_disparity, max_disparity]


class ProposalAggregator(nn.Module):
    def __init__(self, max_disp, in_planes, batch_norm=True):
        super(ProposalAggregator, self).__init__()

        self.max_disp = max_disp
        self.in_planes = in_planes
        self.batch_norm = batch_norm

        self.conv = conv_bn_relu(batch_norm, in_planes, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.res = Hourglass2D(in_planes=48, batch_norm=batch_norm)
        self.lastConv = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

        self.disp_predictor = FasterSoftArgmin(max_disp=max_disp)

    def forward(self, raw_cost):
        cost = self.conv(raw_cost)
        res_cost, _, _ = self.res(cost)
        cost = cost + res_cost
        cost = self.lastConv(cost)

        disp = self.disp_predictor(cost)

        return disp, cost


class MonoStereoSampler(nn.Module):
    def __init__(self, max_disp,
                 disparity_sample_number=4,
                 scale=4,
                 in_planes=32,
                 C=16,
                 batch_norm=True):

        super(MonoStereoSampler, self).__init__()

        self.max_disp = max_disp
        self.batch_norm = batch_norm

        self.scale = scale
        self.disparity_sample_number = disparity_sample_number

        self.in_planes = in_planes
        self.agg_planes = max_disp//scale
        self.C = C

        self.correlation = correlation1d_cost

        self.proposal_aggregator = ProposalAggregator(max_disp//scale, self.agg_planes, batch_norm=batch_norm)

        self.deformable_sampler = DeformOffsetNet(in_planes=1+in_planes+self.agg_planes,
                                                  disparity_sample_number=disparity_sample_number,
                                                  kernel_size=3, stride=1, padding=1, dilation=1, bias=False,
                                                  C=C, batch_norm=batch_norm)

    def get_proposal(self, left, right):

        raw_cost = self.correlation(left, right, max_disp=self.max_disp//4)

        proposal_disp, proposal_cost = self.proposal_aggregator(raw_cost)

        return [proposal_disp], [proposal_cost]


    def forward(self, left, right, proposal_disp, proposal_cost, confidence_cost):

        offset_context = torch.cat((proposal_disp, proposal_cost, left), dim=1)
        disparity_sample, offsets, masks, bound_disps  = self.deformable_sampler(offset_context, proposal_disp, confidence_cost)

        return disparity_sample, offsets, masks, bound_disps


