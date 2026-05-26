#!/bin/bash
#SBATCH -J rcmd-general
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Generalized RC-MD slurm script for ic5c02384 reactions.
#
# Usage:
#   REACTION=1-Me__CO2 TARGET=product RC_K=0.006 MAX_STEP=1000 sbatch run_rcmd.slurm
#
# Environment variables:
#   REACTION       reaction directory name (e.g. 1-Me__CO2)
#   MODE           bias mode (default: rc-md)
#   TARGET         ts or product (default: product)
#   RC_K           harmonic force constant in Ha/Bohr^2 (default: 0.006)
#   MAX_STEP       number of MD steps (default: 1000)
#   TEMP_BATH      thermostat temperature in K (default: 300)
#   INITIAL_TEMP   initial velocity temperature in K (default: 300)
#   SEED           random seed (default: 0)
#   DT             timestep in fs (default: 0.5)
#   MODEL          path to DeePMD model

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-/home/lhshen/RTIP/rtipmd/jax}}"
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

# ----- parameters -----
JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}"
REACTION="${REACTION:?REACTION env var required (e.g. 1-Me__CO2)}"
MODE="${MODE:-rc-md}"
TARGET="${TARGET:-product}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
MAX_STEP="${MAX_STEP:-1000}"
RC_K="${RC_K:-0.006}"
DT="${DT:-0.5}"
TEMP_BATH="${TEMP_BATH:-300.0}"
INITIAL_TEMP="${INITIAL_TEMP:-300.0}"
SEED="${SEED:-0}"
SYNTH_DIST="${SYNTH_DIST:-5.0}"

REACTION_DIR="examples/ic5c02384/reactions/${REACTION}"
INPUT="examples/ic5c02384/reactions/${REACTION}/IS.xyz"

# Auto-generate output prefix
k_int=$(awk "BEGIN {printf \"%04d\", ${RC_K}*1000}")
temp_int=$(awk "BEGIN {printf \"%d\", ${TEMP_BATH}}")
OUTPUT_PREFIX="${OUTPUT_PREFIX:-bias_${REACTION}_${MODE}_${TARGET}_k${k_int}_T${temp_int}_${MAX_STEP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PREFIX}}"

if [[ ! -f "${MODEL}" ]]; then
    echo "ERROR: DeepMD model not found: ${MODEL}" >&2
    exit 1
fi

if [[ ! -d "${REACTION_DIR}" ]]; then
    echo "ERROR: reaction directory not found: ${REACTION_DIR}" >&2
    exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "WORKDIR=${WORKDIR}"
echo "REACTION=${REACTION}"
echo "MODE=${MODE}"
echo "TARGET=${TARGET}"
echo "INPUT=${INPUT}"
echo "REACTION_DIR=${REACTION_DIR}"
echo "MODEL=${MODEL}"
echo "MAX_STEP=${MAX_STEP}"
echo "RC_K=${RC_K}"
echo "DT=${DT}"
echo "TEMP_BATH=${TEMP_BATH}"
echo "INITIAL_TEMP=${INITIAL_TEMP}"
echo "SEED=${SEED}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"

# Verify Python environment
echo "python=$(command -v python)"
python - <<'PY'
import deepmd, jax
print("deepmd", getattr(deepmd, "__version__", "unknown"))
print("jax", jax.__version__)
print("jax_enable_x64", jax.config.read("jax_enable_x64"))
print("jax_devices", jax.devices())
PY

declare -a RUN_ARGS=(
    --mode "${MODE}"
    --target "${TARGET}"
    --reaction-dir "${REACTION_DIR}"
    --input "${INPUT}"
    --model "${MODEL}"
    --output-dir "${OUTPUT_DIR}"
    --output-prefix "${OUTPUT_PREFIX}"
    --max-step "${MAX_STEP}"
    --rc-k "${RC_K}"
    --dt "${DT}"
    --temp-bath "${TEMP_BATH}"
    --initial-temp "${INITIAL_TEMP}"
    --seed "${SEED}"
    --synth-dist "${SYNTH_DIST}"
)

python examples/ic5c02384/run_rcmd.py "${RUN_ARGS[@]}"

echo "Outputs:"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.pdb"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.out"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.md"
echo "Done."
