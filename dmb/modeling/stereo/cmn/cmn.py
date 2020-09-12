import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.layers.basic_layers import conv_bn_relu

from .loss import make_cmn_loss_evaluator


class ConfHead(nn.Module):
    """
    Args:
        in_planes (int): usually cost volume used to calculate confidence map with $in_planes$ in Channel Dimension
        batch_norm, (bool): whether use batch normalization layer, default True
    Inputs:
        cost, (tensor): cost volume in (BatchSize, in_planes, Height, Width) layout
    Outputs:
        confCost, (tensor): in (BatchSize, 1, Height, Width) layout
    """

    def __init__(self, in_planes, batch_norm=True):
        super(ConfHead, self).__init__()

        self.in_planes = in_planes
        self.sec_in_planes = int(self.in_planes // 3)
        self.sec_in_planes = self.sec_in_planes if self.sec_in_planes > 0 else 1

        self.conf_net = nn.Sequential(
            conv_bn_relu(batch_norm, self.in_planes, self.sec_in_planes, 3, 1, 1, bias=False),
            nn.Conv2d(self.sec_in_planes, 1, 1, 1, 0, bias=False)
        )

    def forward(self, cost):
        conf = self.conf_net(cost)
        return conf


# confidence measure network
class Cmn(nn.Module):

    def __init__(self, cfg, in_planes, num, alpha, beta):
        super(Cmn, self).__init__()
        self.cfg = cfg.copy()

        batch_norm = self.cfg.model.batch_norm
        conf_heads = nn.ModuleList(
            [ConfHead(in_planes, batch_norm) for _ in range(num)]
        )
        loss_evaluator = make_cmn_loss_evaluator(cfg)

        self.alpha = alpha
        self.beta = beta

        self.conf_heads = conf_heads
        self.loss_evaluator = loss_evaluator

    def get_confidence(self, costs):
        assert len(self.conf_heads) == len(costs), "NUM of confidence heads({}) must be equal to NUM" \
                                                   "of cost volumes({})".format(len(self.conf_heads), len(costs))

        # for convenience to use log sigmoid when calculate loss,
        # we don't directly confidence cost to confidence by sigmoid
        conf_costs = [conf_head(cost) for cost, conf_head in zip(costs, self.conf_heads)]
        # convert to confidence
        confs = [torch.sigmoid(conf_cost) for conf_cost in conf_costs]
        # calculate variance modulated by confidence
        cost_vars = [self.alpha * (1 - conf) + self.beta for conf in confs]

        return confs, cost_vars, conf_costs

    def get_loss(self, confs, target=None):
        cm_losses = self.loss_evaluator(confs, target)

        return cm_losses

    def forward(self, costs, target=None):
        confs, cost_vars, conf_costs = self.get_confidence(costs)

        if self.training:
            cm_losses = self.get_loss(conf_costs, target)
            return cost_vars, cm_losses
        else:
            return cost_vars, confs


def build_cmn(cfg):
    in_planes = cfg.model.cmn.in_planes
    num = cfg.model.cmn.num
    alpha = cfg.model.cmn.alpha
    beta = cfg.model.cmn.beta

    return Cmn(cfg, in_planes, num, alpha, beta)
