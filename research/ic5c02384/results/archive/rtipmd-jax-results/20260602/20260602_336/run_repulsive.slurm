#!/bin/bash
#SBATCH -J rtip-repulsive
#SBATCH -p main
#SBATCH -N 1 -n 1 -c 10 --gres=gpu:1 --time=24:00:00
#SBATCH -o repulsive_path/slurm-%j.out -e repulsive_path/slurm-%j.err

set -euo pipefail

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u

export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

MODEL=/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt
INPUT=repulsive_path/IS.xyz
OUTPUT_DIR=repulsive_path

echo "=== RTIP Repulsive Pathway Sampling ==="
python -m rtip_jax.cli deepmd-pathway \
    --input "${INPUT}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --method rtip

echo "Done."
ls -la "${OUTPUT_DIR}/"
