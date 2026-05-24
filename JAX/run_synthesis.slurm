#!/bin/bash
#SBATCH -J rtip-synthesis
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u
export PYTHONPATH=/home/lhshen/RTIP/JAX/src${PYTHONPATH:+:${PYTHONPATH}}
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

JOB_ID="${SLURM_JOB_ID:-manual}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
INPUT="${INPUT:-IS.xyz}"
OUTPUT_DIR="${OUTPUT_DIR:-${JOB_ID}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${JOB_ID}}"
MAX_STEP="${MAX_STEP:-100}"

mkdir -p "${OUTPUT_DIR}"
echo "JOB_ID=${JOB_ID}"
echo "INPUT=${INPUT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL=${MODEL}"
echo "MAX_STEP=${MAX_STEP}"

python /home/lhshen/RTIP/JAX/run_synthesis_pathway.py "${INPUT}" "${OUTPUT_DIR}" "${OUTPUT_PREFIX}"
echo "Done."
