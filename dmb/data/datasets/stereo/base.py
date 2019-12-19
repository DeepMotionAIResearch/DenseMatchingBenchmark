import json
import os.path as osp
import numpy as np

from torch.utils.data import Dataset


class StereoDatasetBase(Dataset):
    def __init__(self, annFile, root, transform=None):
        self.annFile = annFile
        self.root = root
        self.data_list = self.annLoader()

        # transforms for data augmentation
        self.transform = transform

        self.flag = np.zeros(len(self.data_list), dtype=np.int64)

    def annLoader(self):
        data_list = []
        with open(file=self.annFile, mode='r') as fp:
            data_list.extend(json.load(fp))
        return data_list

    def Loader(self, item):
        raise NotImplementedError

    def __getitem__(self, idx):
        item = self.data_list[idx]
        sample = self.Loader(item)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data_list)

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Root: {}\n'.format(self.root)
        repr_str += ' ' * 4 + 'annFile: {}\n'.format(self.annFile)
        repr_str += ' ' * 4 + 'Length: {}\n'.format(self.__len__())

        return repr_str

    @property
    def name(self):
        raise NotImplementedError
