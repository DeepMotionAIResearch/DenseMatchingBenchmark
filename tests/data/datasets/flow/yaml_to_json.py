import yaml
import os.path as osp
import json
from tqdm import tqdm

if __name__ == '__main__':
    type = 'FlyingChairs'
    root = '/home/youmin/data/OpticalFlow/{}/'.format(type)
    annFile = '/home/youmin/data/annotations/{}/flyingchairs_val.yml'.format(type)
    saveFile = '/home/youmin/data/annotations/{}/eval.json'.format(type)
    data_list = []
    with open(file=annFile, mode='r') as fp:
        data_list.extend(yaml.load(fp, Loader=yaml.BaseLoader))

    Metas = []
    for idx in range(len(data_list)):
        item = data_list[idx]
        meta = dict(
            left_image_path = item[0],
            right_image_path = item[1],
        )
        if len(item) > 2:
            meta.update(flow_path=item[2])
        Metas.append(meta)

    for meta in tqdm(Metas):
        for k, v in meta.items():
            assert osp.exists(osp.join(root, v)), 'Metas:{} not exists'.format(v)

    info_str = '{} Dataset contains:\n' \
               '    {:5d} samples'.format(type, len(Metas))
    print(info_str)

    print('Save to {}'.format(saveFile))
    with open(file=saveFile, mode='w') as fp:
        json.dump(Metas, fp=fp)
