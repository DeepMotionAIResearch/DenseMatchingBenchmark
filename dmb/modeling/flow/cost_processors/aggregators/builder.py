from .hm_net import HMAggregator

AGGREGATORS = {
    'HMNet': HMAggregator,
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
