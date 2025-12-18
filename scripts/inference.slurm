#!/usr/bin/env bash

# This script shows how you can run inference through Slurm.

#SBATCH --job-name=automaldesc-inference

set -ex

# Run the inference script.
python3 -m automaldesc.inference \
    --config_path configs/inference.yaml \
    --tensor_parallel_size "${SLURM_GPUS_PER_NODE}"
