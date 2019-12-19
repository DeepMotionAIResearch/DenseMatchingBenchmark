from .gc import GCAggregator
from .psm import PSMAggregator
from .acf import AcfAggregator

AGGREGATORS = {
    "GC": GCAggregator,
    "PSM": PSMAggregator,
    "ACF": AcfAggregator,
}


def build_cost_aggregator(cfg):
    aggg_type = cfg.model.cost_processor.cost_aggregator.type
    assert aggg_type in AGGREGATORS, "cost_aggregator type not found, excepted: {}," \
                                     "but got {}".format(AGGREGATORS.keys(), aggg_type)

    args = dict(
        max_disp=cfg.model.max_disp,
        in_planes=cfg.model.cost_processor.cost_aggregator.in_planes,
        batch_norm=cfg.model.batch_norm
    )
    aggregator = AGGREGATORS[aggg_type](**args)

    return aggregator
