from .gc import GCAggregator
from .psm import PSMAggregator
from .acf import AcfAggregator, Acf2DAggregator
from .stereonet import StereoNetAggregator
from .deeppruner import DeepPrunerAggregator
from .anynet import AnyNetAggregator
from .monostereo import MonoStereoAggregator

AGGREGATORS = {
    "GC": GCAggregator,
    "PSM": PSMAggregator,
    "ACF": AcfAggregator,
    "ACF2D": Acf2DAggregator,
    'STEREONET': StereoNetAggregator,
    'DEEPPRUNER': DeepPrunerAggregator,
    'ANYNET': AnyNetAggregator,
    'MONOSTEREO': MonoStereoAggregator,
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
