from .faster_soft_argmin import FasterSoftArgmin
from .local_soft_argmin import LocalSoftArgmin
from .soft_argmin import SoftArgmin

PREDICTORS = {
    'DEFAULT': SoftArgmin,
    'FASTER': FasterSoftArgmin,
    'LOCAL': LocalSoftArgmin,
}


def build_disp_predictor(cfg):
    pred_type = cfg.model.disp_predictor.get('type', 'FASTER')

    assert pred_type in PREDICTORS, 'disparity predictor type not found, expected: {},' \
                                    'but got {}'.format(PREDICTORS.keys(), pred_type)

    default_args = cfg.model.disp_predictor.copy()
    default_args.pop('type')

    disp_predictor = PREDICTORS[pred_type](**default_args)

    return disp_predictor
