from .general_stereo_model import GeneralizedStereoModel
from .DeepPruner import DeepPruner

_META_ARCHITECTURES = {
    "GeneralizedStereoModel": GeneralizedStereoModel,
    "DeepPruner": DeepPruner
}


def build_stereo_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
