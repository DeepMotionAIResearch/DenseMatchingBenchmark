#!/usr/bin/env bash


NGPUS=$1
CFG_PATH=$2
PORT=$3

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NGPUS \
        train.py $CFG_PATH --launcher pytorch --validate --gpus $NGPUS
