from .gc_net import GCNetBackbone
from .psm_net import PSMNetBackbone
from .stereo_net import StereoNetBackbone
from .deep_pruner import DeepPrunerBestBackbone, DeepPrunerFastBackbone
from .any_net import AnyNetBackbone
from .mono_stereo import MonoStereoBackbone

BACKBONES = {
    'GCNet': GCNetBackbone,
    'PSMNet': PSMNetBackbone,
    'StereoNet': StereoNetBackbone,
    'BestDeepPruner': DeepPrunerBestBackbone,
    'FastDeepPruner': DeepPrunerFastBackbone,
    'AnyNet': AnyNetBackbone,
    'MonoStereo': MonoStereoBackbone,
}

def build_backbone(cfg):
    backbone_type = cfg.model.backbone.type

    assert backbone_type in BACKBONES, \
        "model backbone type not found, excepted: {}," \
                        "but got {}".format(BACKBONES.keys, backbone_type)

    default_args = cfg.model.backbone.copy()
    default_args.pop('type')
    default_args.update(batch_norm=cfg.model.batch_norm)

    backbone = BACKBONES[backbone_type](**default_args)

    return backbone
