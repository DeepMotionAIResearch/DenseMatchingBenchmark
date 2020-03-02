import torch
import torch.nn as nn
import torch.nn.functional as F

from dmb.modeling.flow.backbones import build_backbone
from dmb.modeling.flow.cost_processors import build_cost_processor
from dmb.modeling.flow.flow_refinement import build_flow_refinement
from dmb.modeling.flow.losses import make_gof_loss_evaluator


class HMNet(nn.Module):
    def __init__(self, cfg):
        super(HMNet, self).__init__()
        self.cfg = cfg.copy()

        self.backbone = build_backbone(cfg)

        # cost aggregation
        self.cost_processor = build_cost_processor(cfg)

        self.flow_refinement = None
        if 'flow_refinement' in cfg.model:
            self.flow_refinement = build_flow_refinement(cfg)

        self.loss_evaluator = make_gof_loss_evaluator(cfg)

    def forward(self, batch):
        # parse batch
        # [B, 3, H, W]
        ref_img, tgt_img = batch['leftImage'], batch['rightImage']
        target = batch['flow'] if 'flow' in batch else None

        ref_fms, tgt_fms, global_fms = self.backbone(ref_img, tgt_img)

        # [B, 32, H//4, W//4], [B, 64, H//8, W//8], [B, 96, H//16, W//16],
        # [B, 128, H//32, W//32], [B, 196, H//64, W//64]
        ref_fms_4, ref_fms_8, ref_fms_16, ref_fms_32, ref_fms_64 = ref_fms
        tgt_fms_4, tgt_fms_8, tgt_fms_16, tgt_fms_32, tgt_fms_64 = tgt_fms
        global_fms_4, global_fms_8, global_fms_16, global_fms_32, global_fms_64 = global_fms


        # cost aggregation

        # initial guess stage at resolution 1//64
        # [B, 2, H//64, W//64], [B, 2, H//64, W//64], [B, 2, H//32, W//32], [B, 2, H//32, W//32]
        flow_64, cost_64, up_flow_64, up_cost_64, global_flow_64 = self.cost_processor(
            stage='init_guess', left=ref_fms_64, right=tgt_fms_64, global_cost=global_fms_64,
            last_stage_cost=None, last_stage_flow=None)

        # coarse-to-fine, warp at resolution 1/32 and refine the residual
        #[B, 2, H//32, W//32], [B, 2, H//32, W//32], [B, 2, H//16, W//16], [B, 2, H//16, W//16]
        flow_32, cost_32, up_flow_32, up_cost_32, global_flow_32 = self.cost_processor(
            stage='warp_level_32', left=ref_fms_32, right=tgt_fms_32, global_cost=global_fms_32,
            last_stage_cost=up_cost_64, last_stage_flow=up_flow_64)

        # coarse-to-fine, warp at resolution 1/16 and refine the residual
        # [B, 2, H//16, W//16], [B, 2, H//16, W//16], [B, 2, H//8, W//8], [B, 2, H//8, W//8]
        flow_16, cost_16, up_flow_16, up_cost_16, global_flow_16 = self.cost_processor(
            stage='warp_level_16', left=ref_fms_16, right=tgt_fms_16, global_cost=global_fms_16,
            last_stage_cost=up_cost_32, last_stage_flow=up_flow_32)

        # coarse-to-fine, warp at resolution 1/8 and refine the residual
        # [B, 2, H//8, W//8], [B, 2, H//8, W//8][B, 2, H//4, W//4], [B, 2, H//4, W//4]
        flow_8, cost_8, up_flow_8, up_cost_8, global_flow_8 = self.cost_processor(
            stage='warp_level_8', left=ref_fms_8, right=tgt_fms_8, global_cost=global_fms_8,
            last_stage_cost=up_cost_16, last_stage_flow=up_flow_16)

        # coarse-to-fine, warp at resolution 1/4 and refine the residual
        # [B, 2, H//4, W//4], [B, 2, H//4, W//4], [B, 2, H//2, W//2], [B, 2, H//2, W//2]
        flow_4, cost_4, up_flow_4, up_cost_4, global_flow_4 = self.cost_processor(
            stage='warp_level_4', left=ref_fms_4, right=tgt_fms_4, global_cost=global_fms_4,
            last_stage_cost=up_cost_8, last_stage_flow=up_flow_8)

        # up-sample flow predicted from global cost to full resolution
        global_flows = [global_flow_4, global_flow_8, global_flow_16, global_flow_32, global_flow_64]
        # if any flow map in global flows is none, all flow maps will not be evaluated
        if any([f is None for f in global_flows]):
            global_flows = []
        H, W = ref_img.shape[-2:]
        global_flows = [F.interpolate(f * W / f.shape[-1], size=(H, W), mode='bilinear', align_corners=False) for f in global_flows]

        # flows = [flow_4, flow_8, flow_16, flow_32, flow_64]
        flows = [flow_4, up_flow_8, up_flow_16, up_flow_32, up_flow_64]

        if self.flow_refinement is not None:
            # flow refinement using cost volume
            # [B, 2, H//4 , W//4]
            flow_refine = self.flow_refinement(cost_4, flow_4)
            flows.insert(0, flow_refine)

        # up-sample all flow map to full resolution, length = 9 or 10
        H, W = ref_img.shape[-2:]
        flows = [F.interpolate(f * W / f.shape[-1], size=(H, W), mode='bilinear', align_corners=False) for f in flows]


        if self.training:

            flows.extend(global_flows)

            loss_dict = {}

            loss_args = {}

            gsm_loss_dict = self.loss_evaluator(flows, target, **loss_args)
            loss_dict.update(gsm_loss_dict)

            return {}, loss_dict

        else:

            # visualize residual flow map
            res_flows = []
            for i in range(1, len(flows)):
                res_flows.append(flows[i-1] - flows[i])

            flows.extend(global_flows)
            flows.extend(res_flows)

            results = dict(
                flows=flows,
            )

            return results, {}
