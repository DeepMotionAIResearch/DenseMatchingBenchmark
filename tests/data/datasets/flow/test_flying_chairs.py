import numpy as np
import torch
import unittest

from mmcv import Config

from dmb.data.datasets.flow.builder import build_flow_dataset as build_dataset


class TestFlyingChairsDataset(unittest.TestCase):

    def setUp(self):
        config = dict(
            data=dict(
                train=dict(
                    type='FlyingChairs',
                    data_root='/home/youmin/data/OpticalFlow/FlyingChairs/',
                    annfile='/home/youmin/data/annotations/FlyingChairs/test.json',
                    input_shape=[256, 448],
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        )
        cfg = Config(config)
        self.dataset = build_dataset(cfg, 'train')

    def test_anno_loader(self):
        print(self.dataset)
        print(self.dataset.data_list[111])

    def test_get_item(self):
        for i in range(10):
            sample = self.dataset[i]
            assert isinstance(sample, dict)
            for k, v in zip(sample.keys(), sample.values()):
                if isinstance(v, torch.Tensor):
                    print(k, ': with shape', v.shape)
                if isinstance(v, (tuple, list)):
                    print(k, ': ', v)
                if v is None:
                    print(k, ' is None')

    # @unittest.skip('just skip')
    def test_all_data(self):
        from tqdm import tqdm
        for idx in tqdm(range(len(self.dataset))):
            try:
                item = self.dataset[idx]
            except ValueError:
                print('Cannot find: {} -> {}'.format(idx, self.dataset.data_list[idx]))


if __name__ == '__main__':
    unittest.main()
