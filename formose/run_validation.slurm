#!/bin/bash
#SBATCH -J textbook-md
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# Logs always go to logs/slurm/ (see AGENTS.md conventions)
#SBATCH -o /home/lhshen/RTIP/logs/slurm/validation-%j.out
#SBATCH -e /home/lhshen/RTIP/logs/slurm/validation-%j.err
#
# 教科书反应验证 RTIP-MD（evolution MD）+ DeePMD。
# 用法:
#   CASE=ch2o_h2 BOX=/abs/path/ch2o_h2.xyz MAX_STEP=2000 sbatch run_validation.slurm
# 输出: formose/runs/validation_${CASE}/  (rtip.out / rtip.pdb / rtip_decreasing_steps)

set -eo pipefail

RTIP_ROOT="${RTIP_ROOT:-/home/lhshen/RTIP}"
CASE="${CASE:?CASE must be set, e.g. ch2o_h2}"
BOX="${BOX:?BOX must be an absolute path to the reactant xyz}"
MAX_STEP="${MAX_STEP:-2000}"
CONFIG="${CONFIG:-${RTIP_ROOT}/formose/para_formose.json}"
WORKDIR="${WORKDIR:-${RTIP_ROOT}/formose/runs/validation_${CASE}}"

mkdir -p "${RTIP_ROOT}/logs/slurm"
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
  --config "${CONFIG}" \
  --max-step "${MAX_STEP}" \
  --output-dir "${WORKDIR}"
