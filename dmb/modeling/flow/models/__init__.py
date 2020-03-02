from .HMNet import HMNet
from .PWCNet import PWCNet

_META_ARCHITECTURES = {
    'HMNet': HMNet,
    'PWCNet': PWCNet,
}


def build_flow_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
