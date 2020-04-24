import os
import os.path as osp
from collections import abc as container_abcs
import pandas as pd

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import mmcv
from mmcv.runner import Hook, obj_from_dict
from mmcv.runner import LogBuffer
from mmcv.parallel import scatter, collate

from dmb.visualization.stereo import ShowConf

from .eval import remove_padding, do_evaluation


def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]

    raise TypeError((error_msg.format(type(tensor))))


def flow_evaluation(cfg, flows, data_gpu):
    ori_size = data_gpu['original_size']
    flows = remove_padding(flows, ori_size)

    # process the ground truth flow map
    data_gpu['flow'] = data_gpu['flow'] if 'flow' in data_gpu else None
    if data_gpu['flow'] is not None:
        data_gpu['flow'] = remove_padding(data_gpu['flow'], ori_size)

    gtFlow = data_gpu['flow']

    # default only evaluate the first flow map
    eval_flow_id = cfg.get('eval_flow_id', [0])
    whole_error_dict = {}

    # process flow metric
    for id in eval_flow_id:
        all_error_dict = do_evaluation(
            flows[id], gtFlow, cfg.data.sparse)

        for key in all_error_dict.keys():
            whole_error_dict['metric_flow_{}/all_'.format(id) + key] = all_error_dict[key]

    return whole_error_dict, data_gpu


def flow_output_evaluation_in_pandas(output_dict):
    processed_dict = {}
    pandas_dict = {}
    for key in output_dict.keys():
        # format value
        val = output_dict[key]

        if isinstance(val, float):
            val = "{:.4f}".format(val)

        if 'metric_flow' in key:  # e.g. 'metric_flow_0/all_epe'
            flow_id = key.split('/')[0]
            area, metric = key.split('/')[1].split('_')

            # each flow contains one pd.DataFrame, area as index, metric as columns
            if flow_id not in pandas_dict.keys():
                pandas_dict[flow_id] = {}
            if area not in pandas_dict[flow_id].keys():
                pandas_dict[flow_id][area] = {}
            pandas_dict[flow_id][area][metric] = val

        else:
            processed_dict[key] = val

    # generate pandas
    for key in pandas_dict:
        processed_dict[key] = pd.DataFrame.from_dict(pandas_dict[key], orient='index')

    return processed_dict


class DistEvalHook(Hook):

    def __init__(self, cfg, dataset, interval=1):
        self.cfg = cfg.copy()
        assert isinstance(dataset, Dataset), \
            "dataset must be a Dataset object, not {}".format(type(dataset))
        self.dataset = dataset
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
                result, _ = runner.model(data_gpu)
                flows = result['flows']

                # evaluation
                whole_error_dict, data_gpu = flow_evaluation(self.cfg, flows, data_gpu)

                result = {
                    'Flow': flows,
                    'GroundTruth': data_gpu['flow'],
                    'Error': whole_error_dict,
                }

            # if result contains image, as the process advanced, the cuda cache explodes soon.
            result = to_cpu(result)

            filter_result = dict()
            filter_result['Error'] = result['Error']

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


class DistFlowEvalHook(DistEvalHook):

    def __init__(self, cfg, dataset, interval=1):
        super(DistFlowEvalHook, self).__init__(cfg, dataset, interval)

    def evaluate(self, runner, results):

        error_log_buffer = LogBuffer()
        for result in results:
            error_log_buffer.update(result['Error'])
        error_log_buffer.average()

        # import to tensor-board
        for key in error_log_buffer.output.keys():
            runner.log_buffer.output[key] = error_log_buffer.output[key]

        # for better visualization, format into pandas
        format_output_dict = flow_output_evaluation_in_pandas(error_log_buffer.output)

        runner.logger.info("Epoch [{}] Evaluation Result: \t".format(runner.epoch + 1))

        log_items = []
        for key, val in format_output_dict.items():
            if isinstance(val, pd.DataFrame):
                log_items.append("\n{}:\n{} \n".format(key, val))
            elif isinstance(val, float):
                val = "{:.4f}".format(val)
                log_items.append("{}: {}".format(key, val))
            else:
                log_items.append("{}: {}".format(key, val))

        log_str = ", ".join(log_items)
        runner.logger.info(log_str)
        runner.log_buffer.ready = True
        error_log_buffer.clear()

