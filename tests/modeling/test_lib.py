import argparse

import torch
from mmcv import Config

from dmb.modeling.stereo.models.general_stereo_model import GeneralizedStereoModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train dense matching benchmark')
    parser.add_argument('--config', help='train config file path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda")
    cfg = Config.fromfile(args.config)
    model = GeneralizedStereoModel(cfg).to(device)
    print(model)

    batch = dict(
        leftImage=torch.zeros((1, 3, 256, 512)).to(device),
        rightImage=torch.zeros((1, 3, 256, 512)).to(device),
        leftDisp=torch.ones((1, 1, 256, 512)).to(device),
    )
    model(batch)
