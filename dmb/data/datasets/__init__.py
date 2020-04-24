from .stereo import build_stereo_dataset
from .stereo import SceneFlowDataset, Kitti2012Dataset, Kitti2015Dataset

from .flow import build_flow_dataset
from .flow import FlyingChairsDataset

def build_dataset(cfg, type):
    task = cfg.get('task', 'stereo')
    if task == 'stereo':
        return build_stereo_dataset(cfg, type)
    elif task == 'flow':
        return build_flow_dataset(cfg, type)
