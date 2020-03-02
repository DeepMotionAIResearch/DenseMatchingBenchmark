from .hm_net import HMPredictor

PREDICTORS = {
    'HMNet': HMPredictor,
}


def build_flow_predictor(cfg):
    pred_type = cfg.model.flow_predictor.get('type', 'HMNet')

    assert pred_type in PREDICTORS, 'flow predictor type not found, expected: {},' \
                                    'but got {}'.format(PREDICTORS.keys(), pred_type)

    default_args = cfg.model.flow_predictor.copy()
    default_args.pop('type')

    flow_predictor = PREDICTORS[pred_type](**default_args)

    return flow_predictor
