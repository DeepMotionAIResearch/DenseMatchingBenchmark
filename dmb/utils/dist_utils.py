from collections import OrderedDict

import torch.distributed as dist
from torch._utils import (
    _flatten_dense_tensors, _unflatten_dense_tensors, _take_tensors
)
from mmcv.runner import OptimizerHook

try:
    from apex import amp
    import apex
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def _all_reduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def all_reduce_grads(model, coalesce=True, bucket_size_mb=-1):
    grads = [
        param.grad.data for param in model.parameters()
        if param.requires_grad and param.grad is not None
    ]

    world_size = dist.get_world_size()
    if coalesce:
        _all_reduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


class DistOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1):
        super(DistOptimizerHook, self).__init__(grad_clip)
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        all_reduce_grads(runner.model, self.coalesce, self.bucket_size_mb)
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()


class DistApexOptimizerHook(OptimizerHook):

    def __init__(self, grad_clip=None, coalesce=True, bucket_size_mb=-1, use_apex=True):
        super(DistApexOptimizerHook, self).__init__(grad_clip)
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.use_apex = use_apex

    def after_train_iter(self, runner):
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_losses:
            scaled_losses.backward()
        all_reduce_grads(runner.model, self.coalesce, self.bucket_size_mb)
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
