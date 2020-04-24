from .smooth_l1_loss import DispSmoothL1Loss
from .gerf_loss import DispGERFLoss
from .stereo_focal_loss import StereoFocalLoss
from .relative_loss import RelativeLoss

# smooth l1 loss
def make_sll_loss_evaluator(cfg):
    max_disp = cfg.model.losses.l1_loss.get('max_disp', None)
    weights = cfg.model.losses.l1_loss.weights
    sparse = cfg.data.sparse

    return DispSmoothL1Loss(
        max_disp=max_disp,
        weights=weights, sparse=sparse
    )

# GERF loss, proposed in StereoNet,
# https://github.com/meteorshowers/StereoNet-ActiveStereoNet/blob/master/utils/utils.py#L20
def make_gerf_loss_evaluator(cfg):
    max_disp = cfg.model.losses.gerf_loss.get('max_disp', None)
    weights = cfg.model.losses.gerf_loss.weights
    sparse = cfg.data.sparse

    return DispGERFLoss(
        max_disp=max_disp,
        weights=weights, sparse=sparse
    )


# stereo focal loss
def make_focal_loss_evaluator(cfg):
    max_disp = cfg.model.losses.focal_loss.get('max_disp', None)
    start_disp = cfg.model.losses.focal_loss.get('start_disp', 0)
    dilation = cfg.model.losses.focal_loss.get('dilation', 1)
    weights = cfg.model.losses.focal_loss.get('weights', None)
    coefficient = cfg.model.losses.focal_loss.get('coefficient', 0.0)
    sparse = cfg.data.sparse

    return StereoFocalLoss(
        max_disp=max_disp, start_disp=start_disp, dilation=dilation, weights=weights,
        focal_coefficient=coefficient, sparse=sparse
    )

# relative loss
def make_relative_loss_evaluator(cfg):
    max_disp = cfg.model.losses.relative_loss.get('max_disp', None)
    start_disp = cfg.model.losses.relative_loss.get('start_disp', 0)
    weights = cfg.model.losses.relative_loss.get('weights', None)
    sparse = cfg.data.sparse

    return RelativeLoss(max_disp=max_disp, start_disp=start_disp, weights=weights, sparse=sparse)


class CombinedLossEvaluators(object):

    def __init__(self, cfg, loss_evaluators, loss_weights):
        self.cfg = cfg.copy()
        self.loss_evaluators = loss_evaluators
        self.loss_weights = loss_weights

    def __call__(self, disps, costs, target, **kwargs):
        comb_loss_dict = dict()

        for loss_name, loss_evaluator in self.loss_evaluators.items():
            weight = self.loss_weights[loss_name]
            if isinstance(loss_evaluator, DispSmoothL1Loss):
                loss_dict = loss_evaluator(disps, target)
            elif isinstance(loss_evaluator, DispGERFLoss):
                loss_dict = loss_evaluator(disps, target)
            elif isinstance(loss_evaluator, StereoFocalLoss):
                variance = kwargs['variance']
                loss_dict = loss_evaluator(costs, target, variance)
            elif isinstance(loss_evaluator, RelativeLoss):
                loss_dict = loss_evaluator(kwargs['relative_disps'], target, kwargs['relative_labels'])
            else:
                raise ValueError("{} not implemented.".format(loss_name))

            loss_dict = {k: v * weight for k, v in loss_dict.items()}
            comb_loss_dict.update(loss_dict)

        return comb_loss_dict


# general stereo matching loss
def make_gsm_loss_evaluator(cfg):
    loss_evaluators = dict()
    loss_weights = dict()

    if "l1_loss" in cfg.model.losses:
        l1_loss_evaluator = make_sll_loss_evaluator(cfg)
        loss_evaluators["l1_loss"] = l1_loss_evaluator
        loss_weights["l1_loss"] = cfg.model.losses.l1_loss.weight

    if 'gerf_loss' in cfg.model.losses:
        gerf_loss_evaluator = make_gerf_loss_evaluator(cfg)
        loss_evaluators['gerf_loss'] = gerf_loss_evaluator
        loss_weights['gerf_loss'] = cfg.model.losses.gerf_loss.weight

    if "focal_loss" in cfg.model.losses:
        focal_loss_evaluator = make_focal_loss_evaluator(cfg)
        loss_evaluators["focal_loss"] = focal_loss_evaluator
        loss_weights["focal_loss"] = cfg.model.losses.focal_loss.weight

    if "relative_loss" in cfg.model.losses:
        relative_loss_evaluators = make_relative_loss_evaluator(cfg)
        loss_evaluators["relative_loss"] = relative_loss_evaluators
        loss_weights["relative_loss"] = cfg.model.losses.relative_loss.weight

    return CombinedLossEvaluators(cfg, loss_evaluators, loss_weights)
