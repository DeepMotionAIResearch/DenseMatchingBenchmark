from .GCNet import GCAggregator
from .PSMNet import PSMAggregator
from .AcfNet import AcfAggregator
from .StereoNet import StereoNetAggregator
from .DeepPruner import DeepPrunerAggregator
from .AnyNet import AnyNetAggregator

AGGREGATORS = {
    "GCNet": GCAggregator,
    "PSMNet": PSMAggregator,
    "AcfNet": AcfAggregator,
    'StereoNet': StereoNetAggregator,
    'DeepPruner': DeepPrunerAggregator,
    'AnyNet': AnyNetAggregator,
}


def build_cost_aggregator(cfg):
    agg_type = cfg.model.cost_processor.cost_aggregator.type
    assert agg_type in AGGREGATORS, "cost_aggregator type not found, excepted: {}," \
                                     "but got {}".format(AGGREGATORS.keys(), agg_type)

    default_args = cfg.model.cost_processor.cost_aggregator.copy()
    default_args.pop('type')
    default_args.update(batch_norm=cfg.model.batch_norm)

    aggregator = AGGREGATORS[agg_type](**default_args)

    return aggregator
