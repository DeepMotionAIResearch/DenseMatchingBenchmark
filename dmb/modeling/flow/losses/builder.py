from .p_norm_loss import PNormLoss

# p norm loss
def make_pnl_loss_evaluator(cfg):
    p = cfg.model.losses.p_norm_loss.get('p', None)
    epsilon = cfg.model.losses.p_norm_loss.get('epsilon', None)
    weights = cfg.model.losses.p_norm_loss.get('weights', None)
    sparse = cfg.data.sparse

    return PNormLoss(
        p=p, epsilon=epsilon,
        weights=weights, sparse=sparse
    )

class CombinedLossEvaluators(object):

    def __init__(self, cfg, loss_evaluators, loss_weights):
        self.cfg = cfg.copy()
        self.loss_evaluators = loss_evaluators
        self.loss_weights = loss_weights

    def __call__(self, flows, target, **kwargs):
        comb_loss_dict = dict()

        for loss_name, loss_evaluator in self.loss_evaluators.items():
            weight = self.loss_weights[loss_name]
            if isinstance(loss_evaluator, PNormLoss):
                loss_dict = loss_evaluator(flows, target)
            else:
                raise ValueError("{} not implemented.".format(loss_name))

            loss_dict = {k: v * weight for k, v in loss_dict.items()}
            comb_loss_dict.update(loss_dict)

        return comb_loss_dict


# general optical flow loss
def make_gof_loss_evaluator(cfg):
    loss_evaluators = dict()
    loss_weights = dict()

    if "p_norm_loss" in cfg.model.losses:
        p_norm_loss_evaluator = make_pnl_loss_evaluator(cfg)
        loss_evaluators["p_norm_loss"] = p_norm_loss_evaluator
        loss_weights["p_norm_loss"] = cfg.model.losses.p_norm_loss.weight

    return CombinedLossEvaluators(cfg, loss_evaluators, loss_weights)
