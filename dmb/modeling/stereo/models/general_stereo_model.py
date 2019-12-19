import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.cmn import build_cmn
from dmb.modeling.stereo.disp_predictors import build_disp_predictor
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator


class GeneralizedStereoModel(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedStereoModel, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp
        self.scale = cfg.model.backbone.scale

        self.backbone = build_backbone(cfg)
        self.cost_processor = build_cost_processor(cfg)

        # confidence measurement network
        self.cmn = None
        if 'cmn' in cfg.model:
            self.cmn = build_cmn(cfg)

        self.disp_predictor = build_disp_predictor(cfg)
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

    def forward(self, batch):
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        ref_fms, tgt_fms = self.backbone(ref_img, tgt_img)
        costs = self.cost_processor(ref_fms, tgt_fms, int(self.max_disp // self.scale))
        disps = [self.disp_predictor(cost) for cost in costs]

        cost_vars = None
        if self.training:
            loss_dict = dict()
            if self.cmn is not None:
                cost_vars, cm_losses = self.cmn(costs, target)
                loss_dict.update(cm_losses)

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, cost_vars=cost_vars)
            loss_dict.update(gsm_loss_dict)
            return {}, loss_dict
        else:
            results = dict(
                disps=disps,
                costs=costs,
            )

            return results, {}
