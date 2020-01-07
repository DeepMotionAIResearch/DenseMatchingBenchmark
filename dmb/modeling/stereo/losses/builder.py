from .smooth_l1_loss import DispSmoothL1Loss
from .stereo_focal_loss import StereoFocalLoss


# smooth l1 loss
def make_sll_loss_evaluator(cfg):
    max_disp = cfg.model.max_disp
    weights = cfg.model.losses.l1_loss.weights
    sparse = cfg.data.sparse

    return DispSmoothL1Loss(
        max_disp,
        weights=weights, sparse=sparse
    )


# stereo focal loss
def make_focal_loss_evaluator(cfg):
    max_disp = cfg.model.max_disp
    weights = cfg.model.losses.focal_loss.weights
    coefficient = cfg.model.losses.focal_loss.coefficient
    sparse = cfg.data.sparse

    return StereoFocalLoss(
        max_disp, weights=weights,
        focal_coefficient=coefficient, sparse=sparse
    )


class CombinedLossEvaluators(object):

    def __init__(self, cfg, loss_evaluators, loss_weights):
        self.cfg = cfg.copy()
        self.loss_evaluators = loss_evaluators
        self.loss_weights = loss_weights

    def __call__(self, disps, costs, target, cost_vars=None):
        comb_loss_dict = dict()

        for loss_name, loss_evaluator in self.loss_evaluators.items():
            weight = self.loss_weights[loss_name]
            if isinstance(loss_evaluator, DispSmoothL1Loss):
                loss_dict = loss_evaluator(disps, target)
            elif isinstance(loss_evaluator, StereoFocalLoss):
                loss_dict = loss_evaluator(costs, target, cost_vars)
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

    if "focal_loss" in cfg.model.losses:
        focal_loss_evaluator = make_focal_loss_evaluator(cfg)
        loss_evaluators["focal_loss"] = focal_loss_evaluator
        loss_weights["focal_loss"] = cfg.model.losses.focal_loss.weight

    return CombinedLossEvaluators(cfg, loss_evaluators, loss_weights)
