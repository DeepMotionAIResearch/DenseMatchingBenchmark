#!/usr/bin/env bash


NGPUS=$1
CFG_PATH=$2
PORT=$3
SHOW=$4

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port $PORT --nproc_per_node=$NGPUS \
        test.py $CFG_PATH --launcher pytorch --validate --gpus $NGPUS --show $SHOW
