import numpy as np
import torch
import unittest

from mmcv import Config

from dmb.data.datasets.stereo import build_stereo_dataset as build_dataset


class TestKITTIDataset(unittest.TestCase):

    def setUp(self):
        config = dict(
            data=dict(
                test=dict(
                    type='KITTI-2015',
                    data_root='datasets/KITTI-2015/',
                    annfile='datasets/KITTI-2015/annotations/full_eval.json',
                    input_shape=[384, 1248],
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    toRAM=False,
                )
            )
        )
        cfg = Config(config)
        self.dataset = build_dataset(cfg, 'test')

        import pdb
        pdb.set_trace()

    def test_anno_loader(self):
        print(self.dataset)
        print('toRAM: ', self.dataset.toRAM)
        print(self.dataset.data_list[31])

    def test_get_item(self):
        for i in range(10):
            sample = self.dataset[i]
            assert isinstance(sample, dict)
            print("*" * 20)
            print("Before scatter")
            print("*" * 20)
            for k, v in zip(sample.keys(), sample.values()):
                if isinstance(v, torch.Tensor):
                    print(k, ': with shape', v.shape)
                if isinstance(v, (tuple, list)):
                    print(k, ': ', v)
                if v is None:
                    print(k, ' is None')


if __name__ == '__main__':
    unittest.main()
