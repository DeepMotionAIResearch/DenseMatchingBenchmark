#!/usr/bin/env bash


NGPUS=$1
CFG_PATH=$2

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
        train.py $CFG_PATH --launcher pytorch --validate --gpus $NGPUS