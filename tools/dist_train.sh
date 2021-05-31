#!/usr/bin/env bash


NGPUS=$1
CFG_PATH=$2
PORT=$3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NGPUS \
        train.py $CFG_PATH --launcher pytorch --validate --gpus $NGPUS
