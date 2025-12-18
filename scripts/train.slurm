#!/usr/bin/env bash

# This script can help you run a training job in a Slurm environment.

#SBATCH --job-name=automaldesc-training
#SBATCH --ntasks-per-node=1

set -ex

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -i 6000-7000 -n 1)
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_PER_NODE)
export LAUNCHER="accelerate launch \
    --config_file configs/zero3.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    "
export PROGRAM="\
-m automaldesc.train configs/train.yaml
"
export CMD="$LAUNCHER $PROGRAM"
srun --jobid $SLURM_JOBID bash -c "$CMD"
