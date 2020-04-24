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

from .eval import remove_padding, do_evaluation, do_occlusion_evaluation


def to_cpu(tensor):
    error_msg = "Tensor must contain tensors, dicts or lists; found {}"
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, container_abcs.Mapping):
        return {key: to_cpu(tensor[key]) for key in tensor}
    elif isinstance(tensor, container_abcs.Sequence):
        return [to_cpu(samples) for samples in tensor]

    raise TypeError((error_msg.format(type(tensor))))


def disp_evaluation(cfg, disps, data_gpu):

    # remove the padding when data augmentation
    ori_size = data_gpu['original_size']
    disps = remove_padding(disps, ori_size)

    # process the ground truth disparity map
    data_gpu['leftDisp'] = data_gpu['leftDisp'] if 'leftDisp' in data_gpu else None
    if data_gpu['leftDisp'] is not None:
        data_gpu['leftDisp'] = remove_padding(data_gpu['leftDisp'], ori_size)
    data_gpu['rightDisp'] = data_gpu['rightDisp'] if 'rightDisp' in data_gpu else None
    if data_gpu['rightDisp'] is not None:
        data_gpu['rightDisp'] = remove_padding(data_gpu['rightDisp'], ori_size)

    leftDisp = data_gpu['leftDisp']
    rightDisp = data_gpu['rightDisp']

    # default only evaluate the first disparity map
    eval_disparity_id = cfg.get('eval_disparity_id', [0])
    whole_error_dict = {}

    # process disparity metric
    for id in eval_disparity_id:
        all_error_dict = do_evaluation(
            disps[id], leftDisp, cfg.model.eval.lower_bound, cfg.model.eval.upper_bound)

        for key in all_error_dict.keys():
            whole_error_dict['metric_disparity_{}/all_'.format(id) + key] = all_error_dict[key]

        if cfg.model.eval.eval_occlusion and (leftDisp is not None) and (rightDisp is not None):

            noc_occ_error_dict = do_occlusion_evaluation(
                disps[id], leftDisp, rightDisp,
                cfg.model.eval.lower_bound, cfg.model.eval.upper_bound)

            for key in noc_occ_error_dict.keys():
                whole_error_dict['metric_disparity_{}/'.format(id) + key] = noc_occ_error_dict[key]

    return whole_error_dict, data_gpu


def disp_output_evaluation_in_pandas(output_dict):
    processed_dict = {}
    pandas_dict = {}
    for key in output_dict.keys():
        # format value
        val = output_dict[key]

        if isinstance(val, float):
            val = "{:.4f}".format(val)

        if 'metric_disparity' in key:  # e.g. 'metric_disparity_0/all_epe'
            disparity_id = key.split('/')[0]
            area, metric = key.split('/')[1].split('_')

            # each disparity contains one pd.DataFrame, area as index, metric as columns
            if disparity_id not in pandas_dict.keys():
                pandas_dict[disparity_id] = {}
            if area not in pandas_dict[disparity_id].keys():
                pandas_dict[disparity_id][area] = {}
            pandas_dict[disparity_id][area][metric] = val

        elif 'metric_confidence' in key: # e.g. 'metric_confidence_0/est_0'
            confidence_id = key.split('/')[0]
            sparse_type, percent = key.split('/')[1].split('_')

            # each confidence contains one pd.DataFrame, sparse_type as index, percent as columns
            if confidence_id not in pandas_dict.keys():
                pandas_dict[confidence_id] = {}
            if sparse_type not in pandas_dict[confidence_id].keys():
                pandas_dict[confidence_id][sparse_type] = {}
            pandas_dict[confidence_id][sparse_type][percent] = val

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
                disps = result['disps']
                costs = result['costs']

                # evaluation
                whole_error_dict, data_gpu = disp_evaluation(self.cfg, disps, data_gpu)

                result = {
                    'Disparity': disps,
                    'GroundTruth': data_gpu['leftDisp'],
                    'Error': whole_error_dict,
                }

                if self.cfg.model.eval.is_cost_return:
                    if self.cfg.model.eval.is_cost_to_cpu:
                        costs = [cost.cpu() for cost in costs]
                    result['Cost'] = costs

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

    def __init__(self, cfg, dataset, interval=1):
        super(DistStereoEvalHook, self).__init__(cfg, dataset, interval)
        self.conf_tool = ShowConf()

    def evaluate(self, runner, results):
        self.eval_conf(runner, results, bins_number=100)

        error_log_buffer = LogBuffer()
        for result in results:
            error_log_buffer.update(result['Error'])
        error_log_buffer.average()

        # import to tensor-board
        for key in error_log_buffer.output.keys():
            runner.log_buffer.output[key] = error_log_buffer.output[key]

        # for better visualization, format into pandas
        format_output_dict = disp_output_evaluation_in_pandas(error_log_buffer.output)

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
