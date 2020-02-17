import torch
from torch.nn import functional as F

from dmb.modeling.stereo.losses.conf_nll_loss import ConfidenceNllLoss


def make_conf_nll_loss_evaluator(cfg):
    default_args = cfg.model.cmn.losses.nll_loss.copy()
    default_args.update(sparse=cfg.data.sparse)
    default_args.pop('weight')

    return ConfidenceNllLoss(**default_args)


class CMNLossEvaluator(object):
    def __init__(self, cfg, loss_evaluators, loss_weights):
        self.cfg = cfg.copy()
        self.loss_evaluators = loss_evaluators
        self.loss_weights = loss_weights

    def __call__(self, confs, target):
        loss_dict = dict()

        for loss_name, loss_evaluator in self.loss_evaluators.items():
            weight = self.loss_weights[loss_name]
            if isinstance(loss_evaluator, ConfidenceNllLoss):
                conf_nll_loss_dict = loss_evaluator(confs, target)
                conf_nll_loss_dict = {k: v * weight for k, v in conf_nll_loss_dict.items()}
                loss_dict.update(conf_nll_loss_dict)
            else:
                raise ValueError("{} not implemented.".format(loss_name))

        return loss_dict


def make_cmn_loss_evaluator(cfg):
    loss_evaluators = dict()
    loss_weights = dict()

    if "nll_loss" in cfg.model.cmn.losses:
        conf_nll_loss_evaluator = make_conf_nll_loss_evaluator(cfg)
        loss_evaluators["conf_nll_loss"] = conf_nll_loss_evaluator
        loss_weights["conf_nll_loss"] = cfg.model.cmn.losses.nll_loss.weight

    return CMNLossEvaluator(
        cfg, loss_evaluators, loss_weights
    )
