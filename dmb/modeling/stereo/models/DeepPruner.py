import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.disp_samplers import build_disp_sampler
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.disp_refinement import build_disp_refinement
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator


class DeepPruner(nn.Module):
    """
    DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
    """
    def __init__(self, cfg):
        super(DeepPruner, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp

        self.backbone = build_backbone(cfg)

        self.disp_sampler = build_disp_sampler(cfg)

        self.cost_processor = build_cost_processor(cfg)

        self.disp_refinement = build_disp_refinement(cfg)

        # make general stereo matching loss evaluator
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

    def forward(self, batch):
        # parse batch
        # [B, 3, H, W]
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        # extract image feature
        ref_fms, tgt_fms = self.backbone(ref_img, tgt_img)

        # [B, 32, H//4, W//4], [B, 32, H//2, W//2]
        ref_fms, low_ref_fms = ref_fms
        tgt_fms, low_tgt_fms = tgt_fms

        # compute cost volume

        # "pre"(Pre-PatchMatch) using patch match as sampler
        # [B, patch_match_disparity_sample_number, H//4, W//4]
        disparity_sample = self.disp_sampler(stage='pre', left=ref_fms, right=tgt_fms)

        output = self.cost_processor(stage='pre', left=ref_fms, right=tgt_fms,
                                     disparity_sample=disparity_sample)

        # [B, 1, H//4, W//4],         [B, patch_match_disparity_sample_number, H//4, W//4]
        min_disparity, max_disparity, min_disparity_feature, max_disparity_feature = output

        # "post"(Post-ConfidenceRangePredictor) using uniform sampler
        # [B, uniform_disparity_sample_number, H//4, W//4]
        disparity_sample = self.disp_sampler(stage='post', left=ref_fms, right=tgt_fms,
                                             min_disparity=min_disparity, max_disparity=max_disparity)
        output = self.cost_processor(stage='post', left=ref_fms, right=tgt_fms,
                                     disparity_sample=disparity_sample,
                                     min_disparity_feature=min_disparity_feature,
                                     max_disparity_feature=max_disparity_feature)

        # [B, 1, H//2, W//2],  [B, uniform_disparity_sample_number, H//2, W//2]
        disparity, disparity_feature = output

        disps = [disparity]
        refine_input = torch.cat((low_ref_fms, disparity_feature, disparity), dim=1)
        disps = self.disp_refinement(disps, refine_input)

        disps.extend([disparity, min_disparity, max_disparity])

        H, W = ref_img.shape[-2:]
        disps = [F.interpolate(d * W / d.shape[-1], size=(H, W), mode='bilinear') for d in disps]

        costs = [None]

        if self.training:
            loss_dict = dict()

            loss_args = dict(
                variance = None,
            )

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:

            results = dict(
                disps=disps,
                costs=costs,
            )

            return results, {}
