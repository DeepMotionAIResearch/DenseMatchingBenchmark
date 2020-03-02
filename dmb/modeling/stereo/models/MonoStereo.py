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

class MonoStereo(nn.Module):
    """
    A general stereo matching model which fits most methods.

    """
    def __init__(self, cfg):
        super(MonoStereo, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp

        self.backbone = build_backbone(cfg)

        self.disp_sampler = build_disp_sampler(cfg)

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

        self.min_max_weight = nn.Sequential(
            nn.Conv2d(65, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch):
        # parse batch
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        # extract image feature
        ref_fms, tgt_fms = self.backbone(ref_img, tgt_img)

        proposal_disps, proposal_costs = self.disp_sampler.get_proposal(left=ref_fms, right=tgt_fms)
        confs, variances, conf_costs = self.cmn.get_confidence(proposal_costs)
        # all returned disparity map are in 1/4 resolution
        disparity_sample, offsets, masks, min_max_disps = self.disp_sampler(left=ref_fms,
                                                                            right=tgt_fms,
                                                                            proposal_disp=proposal_disps[0],
                                                                            proposal_cost=proposal_costs[0],
                                                                            confidence_cost=conf_costs[0])
        # up-sample to full resolution
        h, w = ref_img.shape[-2:]
        ph, pw = disparity_sample.shape[-2:]
        full_disparity_sample = F.interpolate(disparity_sample * w / pw, size=(h, w), mode='bilinear', align_corners=False)
        full_proposal_disps = [F.interpolate(d * w / d.shape[-1], size=(h, w), mode='bilinear', align_corners=False) for d in proposal_disps]
        full_offsets = [F.interpolate(o*w/o.shape[-1], size=(h, w), mode='bilinear', align_corners=False) for o in offsets]
        full_masks = [F.interpolate(m, size=(h, w), mode='bilinear', align_corners=False) for m in masks]
        full_min_max_disps = [F.interpolate(d * w / d.shape[-1], size=(h, w), mode='bilinear', align_corners=False) for d in min_max_disps]

        # get min max loss weight
        weight_context = torch.cat((proposal_costs[0], ref_fms, proposal_disps[0]), dim=1)
        confidence_weight = self.min_max_weight(weight_context) + 1e-5
        confidence_weight = F.interpolate(confidence_weight, size=(h,w), mode='bilinear', align_corners=False)

        # compute cost volume
        costs = self.cost_processor(ref_fms, tgt_fms, disp_sample=disparity_sample)

        # disparity prediction
        disps = [self.disp_predictor(cost, disp_sample=full_disparity_sample) for cost in costs]

        # disparity refinement
        if self.disp_refinement is not None:
            disps = self.disp_refinement(disps, ref_fms, tgt_fms, ref_img, tgt_img)

        # refined, estimated, coarse, min, max
        disps.extend(full_proposal_disps)
        # disps.extend(full_min_max_disps)

        # supervise cost computed in disparity sampler network
        costs = proposal_costs

        if self.training:
            loss_dict = dict()
            if self.cmn is not None:
                # confidence measurement network
                cm_losses = self.cmn.get_loss(confs=confs, target=target)
                loss_dict.update(cm_losses)

            min_disparity, max_disparity = full_min_max_disps
            # for min disparity, computing relative loss to make sure it smaller than min disparity
            min_label =  1 * torch.ones_like(min_disparity).to(min_disparity)
            # for max disparity, computing relative loss to make sure it larger than min disparity
            max_label = -1 * torch.ones_like(max_disparity).to(max_disparity)

            loss_args = dict(
                variance = variances,
                relative_disps = [min_disparity, max_disparity],
                relative_labels = [min_label, max_label],
            )

            mask = (target > 0) & (target < self.cfg.max_disp)
            exp_confidence_weight = torch.exp(-confidence_weight[mask])
            min_loss = F.smooth_l1_loss(min_disparity[mask] * exp_confidence_weight,
                                        target[mask] * exp_confidence_weight, reduction='mean')
            max_loss = F.smooth_l1_loss(max_disparity[mask] * exp_confidence_weight,
                                        target[mask] * exp_confidence_weight, reduction='mean')
            uncertainty_loss = 100 * confidence_weight[mask] / mask.float().sum()

            loss_dict.update(l1_loss_min=min_loss, l1_loss_max=max_loss, uncertainty_loss=uncertainty_loss)

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:
            disps.extend(full_min_max_disps)

            # visualize residual disparity map
            res_disps = []
            for i in range(1, len(disps)):
                res_disps.append((disps[i-1] - disps[i]).abs())
            disps.extend(res_disps)

            results = dict(
                disps=disps,
                costs=costs,
                offsets=full_offsets,
                masks=full_masks,
            )

            if self.cmn is not None:
                confs.append(confidence_weight)
                # confidence measurement network
                results.update(confs=confs)

            return results, {}
