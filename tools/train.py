from __future__ import division

import argparse
import os
import os.path as osp
import time

import torch

from mmcv import Config
from mmcv import mkdir_or_exist

from dmb.utils.collect_env import collect_env_info
from dmb.utils.env import init_dist, get_root_logger, set_random_seed
from dmb.modeling import build_model
from dmb.data.datasets import build_dataset
from dmb.apis.train import train_matcher


def parse_args():
    parser = argparse.ArgumentParser(description='Training dense matching benchmark')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use (only applicable to non-distributed training)'
    )
    parser.add_argument('--seed', type=int, default=1, help='random seed')
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

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.validate is not None:
        cfg.validate = args.validate
    if args.gpus is not None:
        cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    mkdir_or_exist(cfg.work_dir)
    # init logger before other step and setup training logger
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}_log.txt'.format(timestamp))
    logger = get_root_logger(cfg.work_dir, cfg.log_level, filename=log_file)
    logger.info("Using {} GPUs".format(cfg.gpus))
    logger.info('Distributed training: {}'.format(distributed))

    # log environment info
    logger.info("Collecting env info (might take some time)")
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line)
    logger.info("\n" + collect_env_info())
    logger.info('\n' + dash_line)

    logger.info(args)

    logger.info("Running with config:\n{}".format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg)

    train_dataset = build_dataset(cfg, 'train')
    eval_dataset = build_dataset(cfg, 'eval')
    # all data here will be visualized as image on tensorboardX
    vis_dataset = build_dataset(cfg, 'vis')

    if cfg.checkpoint_config is not None:
        # save config file content in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text,
        )

    train_matcher(
        cfg, model, train_dataset,
        eval_dataset, vis_dataset,
        distributed=distributed,
        validate=args.validate,
        logger=logger
    )


if __name__ == '__main__':
    main()
