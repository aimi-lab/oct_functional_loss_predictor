#!/bin/bash
#SBATCH --job-name="GradCam-Inference"
#SBATCH --output=/storage/homefs/ms22q288/logs/ubelix/%j-%x.out

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4

#SBATCH --partition=gpu-invest
#SBATCH --account=invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:rtx3090:1

# Your code below this line
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"

# export CUDA_LAUNCH_BLOCKING=1

# Activate venv
source /storage/homefs/ms22q288/projects/perimetry_project/.venv/bin/activate

# add the launching directory to the python path, required for import of modules
export PYTHONPATH=$PYTHONPATH:$(pwd)


python resnet/oct2vf_cli.py infer \
    --model-dir ./resnet/runs/REGR_PRETRAIN_AUGMENT_20220804-123227__ep100_bs032_lr1.00E-02_MD_ONH_ADAM_RNFL_RESNET18 \
    --out-dir ./grad_cam_experiments/exp_01 \
    --grad-cam