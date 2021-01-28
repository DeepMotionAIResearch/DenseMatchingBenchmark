#!/bin/bash
python demo.py \
    --config-path ../configs/AcfNet/scene_flow_adaptive.py \
    --checkpoint-path /data/exps/AcfNet/scene_flow_adaptive/epoch_20.pth \
    --data-root ./demo_data/ \
    --device cuda:0 \
    --log-dir /data/exps/AcfNet/scene_flow_adaptive/output/ \
    --pad-to-shape 544 960 \
    --scale-factor 1.0 \
    --disp-div-factor 1.0 \
