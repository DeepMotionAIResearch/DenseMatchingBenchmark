import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.disp_samplers import build_disp_sampler
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.cmn import build_cmn
from dmb.modeling.stereo.disp_predictors import build_disp_predictor
from dmb.modeling.stereo.disp_refinement import build_disp_refinement
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator


class GeneralizedStereoModel(nn.Module):
    """
    A general stereo matching model which fits most methods.

    """
    def __init__(self, cfg):
        super(GeneralizedStereoModel, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp

        self.backbone = build_backbone(cfg)

        self.cost_processor = build_cost_processor(cfg)

        # confidence measurement network
        self.cmn = None
        if 'cmn' in cfg.model:
            self.cmn = build_cmn(cfg)

        self.disp_predictor = build_disp_predictor(cfg)

        self.disp_refinement = None
        if 'disp_refinement' in cfg.model:
            self.disp_refinement = build_disp_refinement(cfg)

        # make general stereo matching loss evaluator
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

    def forward(self, batch):
        # parse batch
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        # extract image feature
        ref_fms, tgt_fms = self.backbone(ref_img, tgt_img)

        # compute cost volume
        costs = self.cost_processor(ref_fms, tgt_fms)

        # disparity prediction
        disps = [self.disp_predictor(cost) for cost in costs]

        # disparity refinement
        if self.disp_refinement is not None:
            disps = self.disp_refinement(disps, ref_fms, tgt_fms, ref_img, tgt_img)

        if self.training:
            loss_dict = dict()
            variance = None
            if hasattr(self.cfg.model.losses, 'focal_loss'):
                variance = self.cfg.model.losses.focal_loss.get('variance', None)

            if self.cmn is not None:
                # confidence measurement network
                variance, cm_losses = self.cmn(costs, target)
                loss_dict.update(cm_losses)

            loss_args = dict(
                variance = variance,
            )

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:

            results = dict(
                disps=disps,
                costs=costs,
            )

            if self.cmn is not None:
                # confidence measurement network
                variance, confs = self.cmn(costs, target)
                results.update(confs=confs)

            return results, {}
