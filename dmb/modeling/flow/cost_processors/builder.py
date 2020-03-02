import torch.nn as nn

from .aggregators import build_cost_aggregator

from .hm_net import HMNetProcessor

PROCESSORS = {
    'HMNet': HMNetProcessor,
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

