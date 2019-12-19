import torch.nn as nn

from .utils.cat_fms import CAT_FUNCS
from .aggregators import build_cost_aggregator


class CostProcessor(nn.Module):

    def __init__(self):
        super(CostProcessor, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class CatCostProcessor(CostProcessor):

    def __init__(self, cat_func, aggregator):
        super(CatCostProcessor, self).__init__()
        self.cat_func = cat_func
        self.aggregator = aggregator

    def forward(self, ref_fms, tgt_fms, max_disp):
        # 1. build raw cost by concat
        cat_cost = self.cat_func(ref_fms, tgt_fms, max_disp)

        # 2. aggregate cost by 3D-hourglass
        costs = self.aggregator(cat_cost)

        return costs


def build_cost_processor(cfg):
    cat_func = CAT_FUNCS[cfg.model.cost_processor.cat_func]
    aggregator = build_cost_aggregator(cfg)

    return CatCostProcessor(
        cat_func=cat_func,
        aggregator=aggregator,
    )
