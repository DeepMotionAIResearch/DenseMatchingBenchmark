from dmb.data.transforms import Compose
from dmb.data.transforms import flow_trans as T

from dmb.data.datasets.flow import FlyingChairsDataset


def build_transforms(cfg, type, is_train):
    input_shape = cfg.data[type].input_shape
    mean = cfg.data[type].mean
    std = cfg.data[type].std

    if is_train:
        transform = Compose(
            [
                # T.RandomTranslate(10),
                # T.RandomRotate(angle=5, diff_angle=10),
                T.ToTensor(),
                T.RandomCrop(input_shape),
                # T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.Normalize(mean, std),
                # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            ]
        )
    else:
        transform = Compose(
            [
                T.ToTensor(),
                T.CenterCat(input_shape),
                T.Normalize(mean, std),
            ]
        )

    return transform


def build_flow_dataset(cfg, type):
    if type not in cfg.data:
        return None

    data_root = cfg.data[type].data_root
    data_type = cfg.data[type].type
    annFile = cfg.data[type].annfile

    is_train = True if type == 'train' else False
    transforms = build_transforms(cfg, type, is_train=is_train)

    if 'FlyingChairs' in data_type:
        dataset = FlyingChairsDataset(annFile, data_root, transforms)
    else:
        raise ValueError("invalid data type: {}".format(data_type))

    return dataset
