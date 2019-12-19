from .faster_soft_argmin import FasterSoftArgmin


def build_disp_predictor(cfg):
    max_disp = cfg.model.max_disp
    alpha = cfg.model.disp_predictor.alpha

    disp_predictor = FasterSoftArgmin(
        max_disp, alpha=alpha
    )
    return disp_predictor
