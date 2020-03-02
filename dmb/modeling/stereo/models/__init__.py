from .general_stereo_model import GeneralizedStereoModel
from .DeepPruner import DeepPruner
from .AnyNet import AnyNet
from .MonoStereo import MonoStereo

_META_ARCHITECTURES = {
    "GeneralizedStereoModel": GeneralizedStereoModel,
    "DeepPruner": DeepPruner,
    "AnyNet": AnyNet,
    "MonoStereo": MonoStereo,
}


def build_stereo_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
