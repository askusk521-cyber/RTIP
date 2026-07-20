#!/bin/bash
#SBATCH -J prog-rcmd
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/home/lhshen/RTIP/rtipmd/jax}"
cd "${WORKDIR}"

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-1024}"

JOB_ID="${SLURM_JOB_ID:-manual}"
REACTION="${REACTION:?REACTION required}"
TARGET="${TARGET:-product}"
K_STAGES="${K_STAGES:-0.006,0.01,0.02}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-500}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
TEMP_BATH="${TEMP_BATH:-300.0}"
INITIAL_TEMP="${INITIAL_TEMP:-150.0}"
SEED="${SEED:-0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-prog_${REACTION}_${TARGET}_3stage}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lhshen/RTIP/JAX/${OUTPUT_PREFIX}}"

REACTION_DIR="examples/ic5c02384/reactions/${REACTION}"

echo "SLURM_JOB_ID=${JOB_ID}"
echo "REACTION=${REACTION}"
echo "K_STAGES=${K_STAGES}"
echo "STEPS_PER_STAGE=${STEPS_PER_STAGE}"
echo "INITIAL_TEMP=${INITIAL_TEMP}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo "python=$(command -v python)"
python - <<'PY'
import deepmd, jax
print("deepmd", getattr(deepmd, "__version__", "unknown"))
print("jax", jax.__version__)
print("jax_devices", jax.devices())
PY

python examples/ic5c02384/run_progressive_rcmd.py     --reaction-dir "${REACTION_DIR}"     --target "${TARGET}"     --k-stages "${K_STAGES}"     --steps-per-stage "${STEPS_PER_STAGE}"     --temp-bath "${TEMP_BATH}"     --initial-temp "${INITIAL_TEMP}"     --model "${MODEL}"     --output-dir "${OUTPUT_DIR}"     --output-prefix "${OUTPUT_PREFIX}"     --seed "${SEED}"

echo "Done. Outputs in: ${OUTPUT_DIR}"
