from __future__ import division

import re
from collections import OrderedDict

import torch
from mmcv.runner import Hook, Runner, DistSamplerSeedHook, obj_from_dict
from mmcv.runner.hooks import EmptyCacheHook
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from dmb.utils.env import get_root_logger
from dmb.data.loaders import build_data_loader
from dmb.data.datasets.evaluation.stereo import DistStereoEvalHook
from dmb.utils import DistOptimizerHook, DistApexOptimizerHook
from dmb.utils import TensorboardLoggerHook, TextLoggerHook
from dmb.visualization.stereo import DistStereoVisHook

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
    import apex
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


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


def batch_processor(model, data, train_mode):
    if train_mode:
        model.train()
    else:
        model.eval()
    _, losses = model(data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars,
        num_samples=len(data['leftImage'].data)
    )

    return outputs


def train_matcher(
        cfg, model, train_dataset,
        eval_dataset=None, test_dataset=None,
        distributed=False, validate=False, logger=None
):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, train_dataset, cfg, eval_dataset, test_dataset, validate=validate, logger=logger)
    else:
        _non_dist_train(model, train_dataset, cfg, eval_dataset, test_dataset, validate=validate, logger=logger)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim, dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_group = {'params': [param]}
            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, train_dataset, cfg, eval_dataset=None, vis_dataset=None, validate=False, logger=None):
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
    model = MMDistributedDataParallel(model)

    # build runner
    runner = Runner(
        model, batch_processor, optimizer, cfg.work_dir, cfg.log_level, logger
    )
    # register hooks
    if cfg.apex.use_mixed_precision:
        optimizer_config = DistApexOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, log_config=None
    )

    # register self-defined logging hooks
    for info in cfg.log_config['hooks']:
        assert isinstance(info, dict) and 'type' in info
        if info['type'] in ['TensorboardLoggerHook']:
            runner.register_hook(
                TensorboardLoggerHook(interval=cfg.log_config.interval, register_logWithIter_keyword=['loss']),
                priority='VERY_LOW'
            )
        if info['type'] in ['TextLoggerHook']:
            runner.register_hook(
                TextLoggerHook(interval=cfg.log_config.interval, ),
                priority='VERY_LOW'
            )

    runner.register_hook(DistSamplerSeedHook())
    runner.register_hook(
        EmptyCacheHook(before_epoch=True, after_iter=False, after_epoch=True),
        priority='VERY_LOW'
    )

    # register eval hooks
    if validate:
        interval = cfg.get('validate_interval', 1)
        if eval_dataset is not None:
            runner.register_hook(DistStereoEvalHook(eval_dataset, interval))
        if vis_dataset is not None:
            runner.register_hook(DistStereoVisHook(vis_dataset, cfg, interval))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, train_dataset, cfg, eval_dataset=None, vis_dataset=None, validate=False, logger=None):
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
    runner.register_training_hooks(
        cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config, cfg.log_config
    )
    runner.register_hook(
        EmptyCacheHook(before_epoch=True, after_iter=False, after_epoch=True),
        priority='VERY_LOW'
    )

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
