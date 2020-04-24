import torch.nn as nn

from .utils.cat_fms import CAT_FUNCS
from .utils.dif_fms import DIF_FUNCS
from .utils.correlation1d_cost import COR_FUNCS
from .aggregators import build_cost_aggregator

from .DeepPruner import DeepPrunerProcessor
from .AnyNet import AnyNetProcessor


class CostProcessor(nn.Module):

    def __init__(self):
        super(CostProcessor, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

# Concatenate left and right feature to form cost volume
class CatCostProcessor(CostProcessor):

    def __init__(self, cfg):
        super(CatCostProcessor, self).__init__()
        cat_func = cfg.model.cost_processor.cost_computation.get('type', 'default')
        self.cat_func = CAT_FUNCS[cat_func]

        self.default_args = cfg.model.cost_processor.cost_computation.copy()
        self.default_args.pop('type')

        self.aggregator = build_cost_aggregator(cfg)

    def forward(self, ref_fms, tgt_fms, disp_sample=None):
        # 1. build raw cost by concat
        cat_cost = self.cat_func(ref_fms, tgt_fms, disp_sample=disp_sample, **self.default_args)

        # 2. aggregate cost by 3D-hourglass
        costs = self.aggregator(cat_cost)

        return costs


# Use the difference between left and right feature to form cost volume
class DifCostProcessor(CostProcessor):

    def __init__(self, cfg):
        super(DifCostProcessor, self).__init__()
        dif_func = cfg.model.cost_processor.cost_computation.get('type', 'default')
        self.dif_func = DIF_FUNCS[dif_func]

        self.default_args = cfg.model.cost_processor.cost_computation.copy()
        self.default_args.pop('type')

        self.aggregator = build_cost_aggregator(cfg)

    def forward(self, ref_fms, tgt_fms, disp_sample=None):
        # 1. build raw cost by concat
        cat_cost = self.dif_func(ref_fms, tgt_fms, disp_sample=disp_sample, **self.default_args)

        # 2. aggregate cost by 3D-hourglass
        costs = self.aggregator(cat_cost)

        return costs


# Use the correlation between left and right feature to form cost volume
class CorCostProcessor(CostProcessor):

    def __init__(self, cfg):
        super(CorCostProcessor, self).__init__()
        cor_func = cfg.model.cost_processor.cost_computation.get('type', 'default')
        self.cor_func = COR_FUNCS[cor_func]

        self.default_args = cfg.model.cost_processor.cost_computation.copy()
        self.default_args.pop('type')

        self.aggregator = build_cost_aggregator(cfg)

    def forward(self, ref_fms, tgt_fms, disp_sample=None):
        # 1. build raw cost by correlation
        cor_cost = self.cor_func(ref_fms, tgt_fms, disp_sample=disp_sample, **self.default_args)

        # 2. aggregate cost by 2D-hourglass
        costs = self.aggregator(cor_cost)

        return costs


PROCESSORS = {
    'Difference': DifCostProcessor,
    'Concatenation': CatCostProcessor,
    'Correlation': CorCostProcessor,
    'DeepPruner': DeepPrunerProcessor,
    'AnyNet': AnyNetProcessor,
}

def build_cost_processor(cfg):
    proc_type = cfg.model.cost_processor.type
    assert proc_type in PROCESSORS, "cost_processor type not found, excepted: {}," \
                                    "but got {}".format(PROCESSORS.keys(), proc_type)

    args = dict(
        cfg=cfg,
    )
    processor = PROCESSORS[proc_type](**args)

    return processor

