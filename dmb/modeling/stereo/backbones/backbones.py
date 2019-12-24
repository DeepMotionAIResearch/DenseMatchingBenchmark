from dmb.modeling.stereo import registry

from .gc_net import GCNetBackbone
from .psm_net import PSMNetBackbone


@registry.BACKBONES.register("GCNet")
def build_gcnet_backbone(cfg):
    in_planes = cfg.model.backbone.in_planes
    batch_norm = cfg.model.batch_norm

    backbone = GCNetBackbone(in_planes, batch_norm)
    return backbone


@registry.BACKBONES.register("PSMNet")
def build_psmnet_backbone(cfg):
    in_planes = cfg.model.backbone.in_planes
    batch_norm = cfg.model.batch_norm

    backbone = PSMNetBackbone(in_planes, batch_norm)
    return backbone


def build_backbone(cfg):
    assert cfg.model.backbone.conv_body in registry.BACKBONES, \
        "cfg.model.backbone.conv_body: {} is not registered in registry".format(
            cfg.model.backbone.conv_body
        )
    return registry.BACKBONES[cfg.model.backbone.conv_body](cfg)
