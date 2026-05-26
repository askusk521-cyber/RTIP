#!/bin/bash
#SBATCH -J bias-1tbu-co2
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Submit from /home/lhshen/RTIP/rtipmd/jax by default.
#
# First-batch examples:
#   MODE=attractive-md TARGET=ts A0=0.0005 TEMP_BATH=300 INITIAL_TEMP=300 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm
#   MODE=attractive-md TARGET=product A0=0.0005 TEMP_BATH=300 INITIAL_TEMP=300 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm
#   MODE=synthesis-md A0=0.002 TEMP_BATH=300 INITIAL_TEMP=300 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm
#   MODE=attractive-pathway TARGET=ts A0=0.0005 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm
#   MODE=attractive-pathway TARGET=product A0=0.0005 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm
#
# Second-batch RC examples:
#   MODE=rc-md TARGET=ts RC_K=0.0005 TEMP_BATH=300 INITIAL_TEMP=300 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm
#   MODE=rc-md TARGET=product RC_K=0.0005 TEMP_BATH=300 INITIAL_TEMP=300 MAX_STEP=200 sbatch run_bias_exploration_1tbu_co2.slurm

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

JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}"
MODE="${MODE:-attractive-md}"
TARGET="${TARGET:-ts}"
INPUT="${INPUT:-examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz}"
REACTION_DIR="${REACTION_DIR:-examples/ic5c02384/reactions/1-tBu__CO2}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
MAX_STEP="${MAX_STEP:-200}"
A0="${A0:-0.0005}"
RC_K="${RC_K:-}"
DT="${DT:-0.5}"
TEMP_BATH="${TEMP_BATH:-300.0}"
INITIAL_TEMP="${INITIAL_TEMP:-300.0}"
SEED="${SEED:-0}"
SYNTH_DIST="${SYNTH_DIST:-5.0}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-bias_1tBu_CO2_${MODE}_${TARGET}_seed${SEED}_step${MAX_STEP}_${JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PREFIX}}"

if [[ ! -f "${MODEL}" ]]; then
    echo "DeepMD model not found: ${MODEL}" >&2
    exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "WORKDIR=${WORKDIR}"
echo "MODE=${MODE}"
echo "TARGET=${TARGET}"
echo "INPUT=${INPUT}"
echo "REACTION_DIR=${REACTION_DIR}"
echo "MODEL=${MODEL}"
echo "MAX_STEP=${MAX_STEP}"
echo "A0=${A0}"
echo "RC_K=${RC_K}"
echo "DT=${DT}"
echo "TEMP_BATH=${TEMP_BATH}"
echo "INITIAL_TEMP=${INITIAL_TEMP}"
echo "SEED=${SEED}"
echo "SYNTH_DIST=${SYNTH_DIST}"
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

declare -a RUN_ARGS=(
    --mode "${MODE}"
    --target "${TARGET}"
    --input "${INPUT}"
    --reaction-dir "${REACTION_DIR}"
    --model "${MODEL}"
    --output-dir "${OUTPUT_DIR}"
    --output-prefix "${OUTPUT_PREFIX}"
    --max-step "${MAX_STEP}"
    --a0 "${A0}"
    --dt "${DT}"
    --temp-bath "${TEMP_BATH}"
    --initial-temp "${INITIAL_TEMP}"
    --seed "${SEED}"
    --synth-dist "${SYNTH_DIST}"
)

if [[ -n "${RC_K}" ]]; then
    RUN_ARGS+=(--rc-k "${RC_K}")
fi

python examples/ic5c02384/run_bias_exploration_1tbu_co2.py "${RUN_ARGS[@]}"

echo "Outputs:"
echo "  ${INPUT}"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.pdb"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.out"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.md"
echo "Done."
