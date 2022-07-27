#!/bin/bash
#( use ## for comments with SBATCH)
## DON'T USE SPACES AFTER COMMAS

# You must specify a valid email address!
#SBATCH --mail-user=davide.scandella@unibe.ch
# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ws_00000

# Job name
#SBATCH --job-name="train_detector"
# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long, gpu-invest

# Runtime and memory
#SBATCH --time=12:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=4G # it's memory PER CPU, NOT TOTAL RAM! maximum RAM is 246G in total
# total RAM is mem-per-cpu * cpus-per-task

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1

# on gpu partition
#SBATCH --gres=gpu:rtx3090:1

# Set the current working directory.
# All relative paths used in the job script are relative to this directory
##SBATCH --workdir=

# create job output file

#SBATCH --output=logs/slurm-%A_%a.out

# For array jobs
# Array job containing 6 tasks, run max 2 tasks at the same time
##SBATCH --array=1-6%2

# param_store=$HOME/oct_biomarker_classification/args.txt
# target=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

# Main Python code below this line
module load Workspace
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate pytorch_env
#module load Workspace
srun python ./oct_cli.py @configs/train.txt
# srun python ./oct_cli.py @configs/infer.txt
