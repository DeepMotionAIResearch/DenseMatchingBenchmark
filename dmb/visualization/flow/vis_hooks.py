import os
import os.path as osp
from collections import abc as container_abcs

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv import mkdir_or_exist
from mmcv.runner import Hook, obj_from_dict
from mmcv.runner import LogBuffer
from mmcv.parallel import scatter, collate

from dmb.visualization.flow.show_result import ShowResultTool
from dmb.data.datasets.evaluation.flow.eval import remove_padding

def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]

    raise TypeError((error_msg.format(type(tensor))))

def prepare_visualize(result, epoch, work_dir, image_name):
    result_tool = ShowResultTool()
    result = result_tool(result)
    mkdir_or_exist(os.path.join(work_dir, image_name))
    save_path = os.path.join(work_dir, image_name, '{}.png'.format(epoch))
    plt.imsave(save_path, result['GroupColor'], cmap=plt.cm.hot)

    log_result = {}
    for pred_item in result.keys():
        log_name = image_name + '/' + pred_item
        if pred_item == 'Flow':
            log_result['image/' + log_name] = result[pred_item]
        if pred_item == 'GroundTruth':
            log_result['image/' + log_name] = result[pred_item]

    return log_result


class DistVisHook(Hook):

    def __init__(self, dataset, cfg, interval=1):
        self.cfg = cfg.copy()
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be a Dataset object, not {}".format(type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        runner.logger.info(
            "Start Visualizing on {} dataset({} images).".format(self.dataset.name, len(self.dataset))
        )

        # get program bar
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        else:
            prog_bar = None

        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()]
            )[0]

            # compute output
            with torch.no_grad():
                ori_result, _ = runner.model(data_gpu)

                # remove the padding when data augmentation
                flows = ori_result['flows']

                ori_size = data_gpu['original_size']
                flows = remove_padding(flows, ori_size)
                target = data_gpu['flow'] if 'flow' in data_gpu else None
                if target is not None:
                    target = remove_padding(target, ori_size)

                result = {
                    'Flow': flows,
                    'GroundTruth': target,
                }


            # convert result to suitable visualization image
            item = self.dataset.data_list[idx]
            result['leftImage'] = imread(
                osp.join(self.cfg.data.vis.data_root, item['left_image_path'])
            ).astype(np.float32)
            result['rightImage'] = imread(
                osp.join(self.cfg.data.vis.data_root, item['right_image_path'])
            ).astype(np.float32)

            image_name = item['left_image_path'].split('/')[-1]
            result = prepare_visualize(result, runner.epoch + 1, self.cfg.work_dir, image_name)

            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, min(runner.world_size, len(self.dataset))):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.visualize(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()

        dist.barrier()
        torch.cuda.empty_cache()

    def visualize(self, runner, results):
        raise NotImplementedError


class DistFlowVisHook(DistVisHook):

    # only log image
    def visualize(self, runner, results):
        for result in results:
            if result is None:
                continue
            for key in result.keys():
                runner.log_buffer.output[key] = result[key]

        # runner.epoch start at 0
        log_str = "Epoch [{}] Visualization Finished!".format(runner.epoch + 1)

        runner.logger.info(log_str)
        runner.log_buffer.ready = True
