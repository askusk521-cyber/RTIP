#!/bin/bash
#SBATCH -J formose-md
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#
# Formose RTIP-MD (evolution MD) with DeePMD as the real PES.
# Usage: sbatch run_formose.slurm   (or SEED=n MAX_STEP=2000 sbatch run_formose.slurm)

set -eo pipefail

RTIP_ROOT="${RTIP_ROOT:-/home/lhshen/RTIP}"
SEED="${SEED:-0}"
MAX_STEP="${MAX_STEP:-10000}"
WORKDIR="${WORKDIR:-${RTIP_ROOT}/formose/runs/seed${SEED}}"
BOX="${BOX:-${RTIP_ROOT}/formose/data/box_seed${SEED}.xyz}"

mkdir -p "${WORKDIR}"

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u
export PYTHONPATH=${RTIP_ROOT}/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export DP_INTRA_OP_PARALLELISM_THREADS="${OMP_NUM_THREADS}"
export DP_INTER_OP_PARALLELISM_THREADS=1

cd "${RTIP_ROOT}"

python -m rtip_jax.cli deepmd-evolution-md \
  --input "${BOX}" \
  --model /home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt \
  --config formose/para_formose.json \
  --max-step "${MAX_STEP}" \
  --output-dir "${WORKDIR}"
