import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.stereo.backbones import build_backbone
from dmb.modeling.stereo.cost_processors import build_cost_processor
from dmb.modeling.stereo.disp_predictors.faster_soft_argmin import FasterSoftArgmin
from dmb.modeling.stereo.disp_refinement import build_disp_refinement
from dmb.modeling.stereo.losses import make_gsm_loss_evaluator


class AnyNet(nn.Module):
    """

    AnyNet: Anytime Stereo Image Depth Estimation on Mobile Devices

    """
    def __init__(self, cfg):
        super(AnyNet, self).__init__()
        self.cfg = cfg.copy()
        self.max_disp = cfg.model.max_disp
        self.stage = cfg.model.stage

        self.backbone = build_backbone(cfg)

        self.cost_processor = build_cost_processor(cfg)

        # disparity predictor
        self.disp_predictor = nn.ModuleDict()
        for st in self.stage:
            self.disp_predictor[st] = FasterSoftArgmin(
                max_disp=cfg.model.disp_predictor.max_disp[st],
                start_disp=cfg.model.disp_predictor.start_disp[st],
                dilation=cfg.model.disp_predictor.dilation[st],
                alpha=cfg.model.disp_predictor.alpha,
                normalize=cfg.model.disp_predictor.normalize,
            )

        self.disp_refinement = build_disp_refinement(cfg)

        # make general stereo matching loss evaluator
        self.loss_evaluator = make_gsm_loss_evaluator(cfg)

    def forward(self, batch):
        # parse batch
        # [B, 3, H, W]
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['leftDisp'] if 'leftDisp' in batch else None

        # extract image feature
        ref_group_fms, tgt_group_fms = self.backbone(ref_img, tgt_img)

        # [B, 8C, H//16, W//16], [B, 4C, H//8, W//8], [B, 2C, H//4, W//4]
        ref_fms_16, ref_fms_8, ref_fms_4 = ref_group_fms
        tgt_fms_16, tgt_fms_8, tgt_fms_4 = tgt_group_fms

        # cost procession

        # initial guess stage with resolution 1/16
        # list, [[B, D, H//16, W//16]]
        costs_init_guess = self.cost_processor(stage='init_guess',
                                               left=ref_fms_16,
                                               right=tgt_fms_16,
                                               disp=None)
        # list, [[B, 1, H//16, W//16]]
        disps_init_guess = [self.disp_predictor['init_guess'](cost) for cost in costs_init_guess]


        # coarse-to-fine, warp at resolution 1/8 and refine the residual
        # list, [[B, D, H//8, W//8]]
        costs_warp_level_8 = self.cost_processor(stage='warp_level_8',
                                                 left=ref_fms_8,
                                                 right=tgt_fms_8,
                                                 disp=disps_init_guess[0])
        # residual disparity map
        # list, [[B, 1, H//8, W//8]]
        disps_warp_level_8 = [self.disp_predictor['warp_level_8'](cost) for cost in costs_warp_level_8]

        # add residual with low resolution disparity map
        high_d = disps_warp_level_8[0]
        low_d = disps_init_guess[0]
        H, W = high_d.shape[-2:]
        h, w = low_d.shape[-2:]
        scale = W / w
        up_low_d = F.interpolate(low_d * scale, size=(H, W), mode='bilinear', align_corners=False)
        disps_warp_level_8 = [up_low_d + high_d]

        # coarse-to-fine, warp at resolution 1/4
        # list, [[B, D, H//4, W//4]]
        costs_warp_level_4 = self.cost_processor(stage='warp_level_4',
                                                 left=ref_fms_4,
                                                 right=tgt_fms_4,
                                                 disp=disps_warp_level_8[0])
        # residual disparity map
        # list, [[B, 1, H//4, W//4]]
        disps_warp_level_4 = [self.disp_predictor['warp_level_4'](cost) for cost in costs_warp_level_4]

        # add residual with low resolution disparity map
        high_d = disps_warp_level_4[0]
        low_d = disps_warp_level_8[0]
        H, W = high_d.shape[-2:]
        h, w = low_d.shape[-2:]
        scale = W / w
        up_low_d = F.interpolate(low_d * scale, size=(H, W), mode='bilinear', align_corners=False)
        disps_warp_level_4 = [up_low_d + high_d]


        # refine the disparity map in `disps_warp_level_4` with spn
        # list, [[B, 1, H//4, W//4],[B, 1, H//4, W//4]]
        disps = disps_warp_level_4
        disps = self.disp_refinement(disps, ref_fms_4, tgt_fms_4, ref_img, tgt_img)

        # list adding, length = 4
        disps = disps + disps_warp_level_8 + disps_init_guess

        # up-sample all disparity map to full resolution, length = 4
        H, W = ref_img.shape[-2:]
        disps = [F.interpolate(d * W / d.shape[-1], size=(H, W), mode='bilinear', align_corners=False) for d in disps]

        # list adding, length = 3
        costs = costs_warp_level_4 + costs_warp_level_8 + costs_init_guess

        if self.training:
            loss_dict = {}

            loss_args = dict(
                variance=None,
            )

            gsm_loss_dict = self.loss_evaluator(disps, costs, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:
            # visualize residual disparity map
            res_disps = []
            for i in range(1, len(disps)):
                res_disps.append(disps[i-1] - disps[i])
            disps.extend(res_disps)

            results = dict(
                disps=disps,
                costs=costs,
            )

            return results, {}
