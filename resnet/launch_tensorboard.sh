#!/bin/bash
module load Workspace
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate pytorch_env
tensorboard --logdir=runs