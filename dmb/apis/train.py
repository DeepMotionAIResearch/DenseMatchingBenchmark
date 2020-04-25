from __future__ import division

import re
from collections import OrderedDict

import torch
from mmcv.runner import Hook, Runner, DistSamplerSeedHook, obj_from_dict
from mmcv.runner.hooks import EmptyCacheHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from dmb.utils.env import get_root_logger
from dmb.data.loaders import build_data_loader
from dmb.utils.solver import build_optimizer
from dmb.data.datasets.evaluation.stereo import DistStereoEvalHook
from dmb.data.datasets.evaluation.flow import DistFlowEvalHook
from dmb.utils import DistOptimizerHook, DistApexOptimizerHook
from dmb.utils import TensorboardLoggerHook, TextLoggerHook
from dmb.visualization.stereo import DistStereoVisHook
from dmb.visualization.flow import DistFlowVisHook

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
    import apex
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def batch_processor(model, data, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    _, losses = model(data)

    def parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    '{} is not a tensor or list of tensors'.format(loss_name))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars

    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars,
        num_samples=len(data['leftImage'].data)
    )

    return outputs


def train_matcher(
        cfg, model, train_dataset,
        eval_dataset=None, vis_dataset=None,
        distributed=False, validate=False, logger=None
):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, train_dataset, cfg, eval_dataset, vis_dataset, validate=validate, logger=logger)
    else:
        _non_dist_train(model, train_dataset, cfg, eval_dataset, vis_dataset, validate=validate, logger=logger)


def _dist_train(
        model, train_dataset, cfg,
        eval_dataset=None, vis_dataset=None, validate=False, logger=None
):
    # prepare data loaders
    data_loaders = [
        build_data_loader(
            train_dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True
        )
    ]
    if cfg.apex.synced_bn:
        # using apex synced BN
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()
    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # Initialize mixed-precision training
    if cfg.apex.use_mixed_precision:
        amp_opt_level = 'O1' if cfg.apex.type == "float16" else 'O0'
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=amp_opt_level, loss_scale=cfg.apex.loss_scale
        )

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters)
    # build runner
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, cfg.log_level, logger
    )

    # register optimizer hooks
    if cfg.apex.use_mixed_precision:
        optimizer_config = DistApexOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    logger.info("Register Optimizer Hook...")
    runner.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config,
        log_config={"interval": cfg.log_config['interval'], 'hooks': []}
    )

    # register self-defined logging hooks
    for info in cfg.log_config['hooks']:
        assert isinstance(info, dict) and 'type' in info
        if info['type'] in ['TensorboardLoggerHook']:
            logger.info("Register Tensorboard Logger Hook...")
            runner.register_hook(
                TensorboardLoggerHook(interval=cfg.log_config.interval, register_logWithIter_keyword=['loss']),
                priority='VERY_LOW'
            )
        if info['type'] in ['TextLoggerHook']:
            logger.info("Register Text Logger Hook...")
            runner.register_hook(
                TextLoggerHook(interval=cfg.log_config.interval, ),
                priority='VERY_LOW'
            )

    logger.info("Register SamplerSeed Hook...")
    runner.register_hook(DistSamplerSeedHook())
    logger.info("Register EmptyCache Hook...")
    runner.register_hook(
        EmptyCacheHook(before_epoch=True, after_iter=False, after_epoch=True),
        priority='VERY_LOW'
    )

    # register eval hooks
    if validate:
        interval = cfg.get('validate_interval', 1)
        task = cfg.get('task', 'stereo')
        if eval_dataset is not None:
            logger.info("Register Evaluation Hook...")
            if task == 'stereo':
                runner.register_hook(DistStereoEvalHook(cfg, eval_dataset, interval))
            elif task == 'flow':
                runner.register_hook(DistFlowEvalHook(cfg, eval_dataset, interval))
        if vis_dataset is not None:
            logger.info("Register Visualization hook...")
            if task == 'stereo':
                runner.register_hook(DistStereoVisHook(vis_dataset, cfg, interval))
            elif task == 'flow':
                runner.register_hook(DistFlowVisHook(vis_dataset, cfg, interval))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(
        model, train_dataset, cfg,
        eval_dataset=None, vis_dataset=None, validate=False, logger=None
):
    # prepare data loaders
    data_loaders = [
        build_data_loader(
            train_dataset,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, cfg.log_level, logger
    )
    logger.info("Register Optimizer Hook...")
    runner.register_training_hooks(
        cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config, cfg.log_config
    )
    logger.info("Register EmptyCache Hook...")
    runner.register_hook(
        EmptyCacheHook(before_epoch=True, after_iter=False, after_epoch=True),
        priority='VERY_LOW'
    )

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
