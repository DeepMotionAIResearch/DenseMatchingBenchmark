from dmb.data.transforms import Compose
from dmb.data.transforms import stereo_trans as T

from dmb.data.datasets.stereo.scene_flow import SceneFlowDataset
from dmb.data.datasets.stereo.kitti import Kitti2012Dataset, Kitti2015Dataset


def build_transforms(cfg, type, is_train):
    input_shape = cfg.data[type].input_shape
    mean = cfg.data[type].mean
    std = cfg.data[type].std

    if is_train:
        transform = Compose(
            [
                T.ToTensor(),
                T.RandomCrop(input_shape),
                T.Normalize(mean, std),
            ]
        )
    else:
        transform = Compose(
            [
                T.ToTensor(),
                T.StereoPad(input_shape),
                T.Normalize(mean, std),
            ]
        )

    return transform


def build_stereo_dataset(cfg, type):
    if type not in cfg.data:
        return None

    data_root = cfg.data[type].data_root
    data_type = cfg.data[type].type
    annFile = cfg.data[type].annfile

    is_train = True if type == 'train' else False
    transforms = build_transforms(cfg, type, is_train=is_train)

    if 'SceneFlow' in data_type:
        dataset = SceneFlowDataset(annFile, data_root, transforms)
    elif 'KITTI' in data_type:
        if '2012' in data_type:
            dataset = Kitti2012Dataset(annFile, data_root, transforms)
        elif '2015' in data_type:
            dataset = Kitti2015Dataset(annFile, data_root, transforms)
        else:
            raise ValueError("invalid data type: {}".format(data_type))
    else:
        raise ValueError("invalid data type: {}".format(data_type))

    return dataset
