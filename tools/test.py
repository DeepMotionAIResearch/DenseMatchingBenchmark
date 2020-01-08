import argparse
import os
import os.path as osp
import shutil
import tempfile

import numpy as np
from imageio import imread

import torch
import torch.distributed as dist

import mmcv
from mmcv import mkdir_or_exist
from mmcv.runner import LogBuffer
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import scatter, collate

from dmb.utils.collect_env import collect_env_info
from dmb.utils.env import init_dist, get_root_logger
from dmb.modeling.stereo import build_stereo_model as build_model
from dmb.data.datasets.stereo import build_dataset
from dmb.apis.inference import save_result
from dmb.visualization.stereo import sparsification_plot


def sparsification_eval(result, cfg):
    if hasattr(cfg, 'sparsification_plot') and cfg.sparsification_plot.doing:
        if 'Confidence' in result and isinstance(result['Confidence'][0], torch.Tensor):
            estConf = result['Confidence'][0].clone()
        if 'GroundTruth' in result and isinstance(result['GroundTruth'], torch.Tensor):
            gtDisp = result['GroundTruth'].clone()
        if 'Disparity' in result and isinstance(result['Disparity'][0], torch.Tensor):
            estDisp = result['Disparity'][0].clone()

        if all([torch.is_tensor(estConf), torch.is_tensor(gtDisp),
                torch.is_tensor(estDisp), isinstance(cfg.sparsification_plot.bins, int)]):
            error_dict = sparsification_plot(
                estDisp, gtDisp, estConf, cfg.sparsification_plot.bins,
                cfg.model.eval.lower_bound, cfg.model.eval.upper_bound)

            return error_dict


def single_gpu_test(model, dataset, cfg, show=False):
    return NotImplementedError


def multi_gpu_test(model, dataset, cfg, show=False, tmpdir=None):
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for idx in range(rank, len(dataset), world_size):
        data = dataset[idx]

        # None type data cannot be scatter, here we pick out the not None type data
        notNoneData = {}
        for k, v in zip(data.keys(), data.values()):
            if v is not None:
                notNoneData[k] = v
        notNoneData = scatter(
            collate([notNoneData], samples_per_gpu=1),
            [torch.cuda.current_device()]
        )[0]

        data.update(notNoneData)

        # TODO: evaluate after generate all predictions!
        with torch.no_grad():
            result, _ = model(data)
            disps = result['disps']

            ori_size = data['original_size']
            target = data_gpu['leftDisp'] if 'leftDisp' in data else None
            target = remove_padding(target, ori_size)
            error_dict = do_evaluation(
                disps[0], target, cfg.model.eval.lower_bound, cfg.model.eval.upper_bound)

            if cfg.model.eval.eval_occlusion and 'leftDisp' in data and 'rightDisp' in data:
                data['leftDisp'] = remove_padding(data['leftDisp'], ori_size)
                data['rightDisp'] = remove_padding(data['rightDisp'], ori_size)

                occ_error_dict = do_occlusion_evaluation(
                    disps[0], data['leftDisp'], data['rightDisp'],
                    cfg.model.eval.lower_bound, cfg.model.eval.upper_bound)
                error_dict.update(occ_error_dict)

            result = {
                'Disparity': disps,
                'GroundTruth': target,
                'Error': error_dict,
            }

        filter_result = {}
        filter_result.update(Error=result['Error'])

        if show:
            item = dataset.data_list[idx]
            result['leftImage'] = imread(
                osp.join(cfg.data.test.data_root, item['left_image_path'])
            ).astype(np.float32)
            result['rightImage'] = imread(
                osp.join(cfg.data.test.data_root, item['right_image_path'])
            ).astype(np.float32)
            image_name = item['left_image_path'].split('/')[-1]
            save_result(result, cfg.out_dir, image_name)

        if hasattr(cfg, 'sparsification_plot'):
            filter_result['Error'].update(sparsification_eval(result, cfg))

        results.append(filter_result)

        if rank == 0:
            batch_size = world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.Tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test dense matching benchmark')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out_dir', help='output result directory')
    parser.add_argument('--show', action='store_true', help='show results in images')
    parser.add_argument('--evaluate', action='store_true', help='whether to evaluate the result')
    parser.add_argument('--gpus', type=int, default=1,
        help='number of gpus to use (only applicable to non-distributed training)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.gpus is not None:
        cfg.gpus = args.gpus
    cfg.show = args.show

    mkdir_or_exist(cfg.out_dir)

    # init logger before other step and setup training logger
    logger = get_root_logger(cfg.out_dir, cfg.log_level, filename="test_log.txt")
    logger.info("Using {} GPUs".format(cfg.gpus))
    logger.info('Distributed training: {}'.format(distributed))

    # log environment info
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info(args)

    logger.info("Running with config:\n{}".format(cfg.text))

    # build the dataset
    test_dataset = build_dataset(cfg, 'test')

    # build the model and load checkpoint
    model = build_model(cfg)
    checkpoint = load_checkpoint(model, cfg.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, test_dataset, cfg, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, test_dataset, cfg, args.show, tmpdir=osp.join(cfg.out_dir, 'temp'))

    rank, _ = get_dist_info()
    if cfg.out_dir is not None and rank == 0:
        result_path = osp.join(cfg.out_dir, 'result.pkl')
        logger.info('\nwriting results to {}'.format(result_path))
        mmcv.dump(outputs, result_path)

        if args.evaluate:
            error_log_buffer = LogBuffer()
            for result in outputs:
                error_log_buffer.update(result['Error'])
            error_log_buffer.average()
            log_items = []
            for key in error_log_buffer.output.keys():

                val = error_log_buffer.output[key]
                if isinstance(val, float):
                    val = '{:.4f}'.format(val)
                log_items.append('{}: {}'.format(key, val))

            if len(error_log_buffer.output) == 0:
                log_items.append('nothing to evaluate!')

            log_str = 'Evaluation Result: \t'
            log_str += ', '.join(log_items)
            logger.info(log_str)
            error_log_buffer.clear()


if __name__ == '__main__':
    main()
