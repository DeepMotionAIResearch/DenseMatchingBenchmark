from .flow.models import _META_ARCHITECTURES as _FLOW_META_ARCHITECTURES
from .stereo.models import _META_ARCHITECTURES as _STEREO_META_ARCHITECTURES

_META_ARCHITECTURES = dict()

_META_ARCHITECTURES.update(_FLOW_META_ARCHITECTURES)
_META_ARCHITECTURES.update(_STEREO_META_ARCHITECTURES)


def build_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
