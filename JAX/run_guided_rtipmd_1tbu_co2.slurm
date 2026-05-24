#!/bin/bash
#SBATCH -J guided-rtipmd
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-/home/lhshen/RTIP/JAX}}"
cd "${WORKDIR}"

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u
export PYTHONPATH=/home/lhshen/RTIP/JAX/src${PYTHONPATH:+:${PYTHONPATH}}

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-1024}"

JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}"
INPUT="${INPUT:-examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz}"
TARGET="${TARGET:-ts}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
MAX_STEP="${MAX_STEP:-200}"
A0="${A0:-0.002}"
DT="${DT:-0.5}"
TEMP_BATH="${TEMP_BATH:-1000.0}"
INITIAL_TEMP="${INITIAL_TEMP:-1000.0}"
OUTPUT_DIR="${OUTPUT_DIR:-guided_rtipmd_1tBu_CO2_${TARGET}_${JOB_ID}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-guided_rtipmd_1tBu_CO2_${TARGET}_${JOB_ID}}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "WORKDIR=${WORKDIR}"
echo "INPUT=${INPUT}"
echo "TARGET=${TARGET}"
echo "MODEL=${MODEL}"
echo "MAX_STEP=${MAX_STEP}"
echo "A0=${A0}"
echo "DT=${DT}"
echo "TEMP_BATH=${TEMP_BATH}"
echo "INITIAL_TEMP=${INITIAL_TEMP}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
echo "python=$(command -v python)"
python - <<'PY'
import deepmd
import jax
print("deepmd", getattr(deepmd, "__version__", "unknown"))
print("jax", jax.__version__)
print("jax_enable_x64", jax.config.read("jax_enable_x64"))
print("jax_devices", jax.devices())
PY

python examples/ic5c02384/run_guided_rtipmd_1tbu_co2.py \
    --input "${INPUT}" \
    --target "${TARGET}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --max-step "${MAX_STEP}" \
    --a0 "${A0}" \
    --dt "${DT}" \
    --temp-bath "${TEMP_BATH}" \
    --initial-temp "${INITIAL_TEMP}"

echo "Outputs:"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.pdb"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.out"
echo "Done."
