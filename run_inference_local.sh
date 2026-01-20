#!/bin/bash


python resnet/oct2vf_cli.py infer \
    --model-dir ./resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-123227__ep100_bs032_lr1.00E-02_MD_ONH_ADAM_RNFL_RESNET18 \
    --out-dir ./grad_cam_experiments/exp_01 \
    --grad-cam