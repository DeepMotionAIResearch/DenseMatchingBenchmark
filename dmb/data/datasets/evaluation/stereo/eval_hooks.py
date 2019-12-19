import os
import os.path as osp
from collections import abc as container_abcs

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import Hook, obj_from_dict
from mmcv.runner import LogBuffer
from mmcv.parallel import scatter, collate

from dmb.visualization.stereo import ShowConf


def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]

    raise TypeError((error_msg.format(type(tensor))))


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError("dataset must be a Dataset object, not {}".format(type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        runner.logger.info(
            "Start evaluation on {} dataset({} images).".format(self.dataset.name, len(self.dataset))
        )
        runner.model.eval()

        # get prog bar
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        else:
            prog_bar = None

        results = [None for _ in range(len(self.dataset))]
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1), [torch.cuda.current_device()]
            )[0]

            # compute output
            with torch.no_grad():
                result = runner.model(data_gpu)
            # if result contains image, as the process advanced, the cuda cache explodes soon.
            result = to_cpu(result)

            filter_result = dict()
            filter_result['Error'] = result['Error']
            if 'Confidence' in result:
                filter_result['Confidence'] = self.process_conf(result, bins_number=100)
            results[idx] = filter_result

            batch_size = runner.world_size

            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, min(runner.world_size, len(self.dataset))):
                tmp_file = osp.join(runner.work_dir, "temp_{}.pkl".format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir, "temp_{}.pkl".format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()
        torch.cuda.empty_cache()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError


class DistStereoEvalHook(DistEvalHook):

    def __init__(self, dataset, interval=1):
        super(DistStereoEvalHook, self).__init__(dataset, interval)
        self.conf_tool = ShowConf()

    def evaluate(self, runner, results):
        self.eval_conf(runner, results, bins_number=100)

        error_log_buffer = LogBuffer()
        for result in results:
            error_log_buffer.update(result['Error'])
        error_log_buffer.average()
        log_items = []
        for key in error_log_buffer.output.keys():
            runner.log_buffer.output[key] = error_log_buffer.output[key]

            val = error_log_buffer.output[key]
            if isinstance(val, float):
                val = "{:.4f}".format(val)
            log_items.append("{}: {}".format(key, val))

        # runner.epoch start at 0
        log_str = "Epoch [{}] Evaluation Result: \t".format(runner.epoch + 1)
        log_str += ", ".join(log_items)
        runner.logger.info(log_str)
        runner.log_buffer.ready = True
        error_log_buffer.clear()

    # confidence distribution statistics
    def process_conf(self, result, bins_number=100):
        if 'Confidence' not in result:
            return

        counts = []
        bin_edges = []
        # for each confidence map, statistic its confidence distribution, and stored in a list
        for i, conf in enumerate(result['Confidence']):
            # hist and bin_edges
            count, bin_edge = self.conf_tool.conf2hist(conf, bins=bins_number)
            counts.append(count)
            bin_edges.append(bin_edge)

        return {
            'counts': counts,
            'bin_edges': bin_edges
        }

    def eval_conf(self, runner, results, bins_number=100):
        # results is a list, corresponds to each test sample,
        # for each sample, the result are saved as dict
        # if the first sample contains the keyword 'Confidence'
        if 'Confidence' not in results[0]:
            return

        # each sample has several confidence map, i.e. bin_edges is a list,
        # with length = confidence map number
        conf_number = len(results[0]['Confidence']['bin_edges'])

        # for each confidence map, statistic its confidence distribution among all samples
        total_counts = np.zeros((conf_number, bins_number))
        total_bin_edges = np.zeros((conf_number, bins_number + 1))
        for result in results:
            # enumerate each sample's every confidence map, and i is the index of confidence map
            for i, conf in enumerate(result['Confidence']['bin_edges']):
                counts, bin_edges = result['Confidence']['counts'][i], result['Confidence']['bin_edges'][i]
                # accumulate each confidence map's counts for all samples
                total_counts[i] = total_counts[i] + counts
                # each confidence map's bin_edges are same
                total_bin_edges[i] = bin_edges

        for i in range(conf_number):
            total_counts[i] = total_counts[i] / sum(total_counts[i])
            name = "figure/confidence_histogram/{}".format(i)
            conf_hist = self.conf_tool.hist2vis(total_counts[i], total_bin_edges[i])
            runner.log_buffer.output[name] = conf_hist

        runner.logger.info("Epoch [{}] Confidence evaluation done!".format(runner.epoch + 1))
        runner.log_buffer.ready = True
