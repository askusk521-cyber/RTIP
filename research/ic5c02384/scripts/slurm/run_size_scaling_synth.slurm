#!/bin/bash
#SBATCH -J synth-sizescale
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Generalized synthesis-MD size_scaling A/B runner.
#
# Submit from /home/lhshen/RTIP/rtipmd/jax (default WORKDIR).
#
# Env overrides:
#   REACTION_DIR (default 1-tBu__CS2), A0, MAX_STEP, TAU, TEMP_BATH,
#   SEED, SIZE_SCALING (0/1), SYNTH_DIST, OUTPUT_DIR, OUTPUT_PREFIX

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-/home/lhshen/RTIP/rtipmd/jax}}"
cd "${WORKDIR}"

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src:/home/lhshen/RTIP/rtipmd/jax/examples/ic5c02384${PYTHONPATH:+:${PYTHONPATH}}

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-1024}"

JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}"
REACTION_DIR="${REACTION_DIR:-examples/ic5c02384/reactions/1-tBu__CS2}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
A0="${A0:-0.0002}"
MAX_STEP="${MAX_STEP:-1500}"
DT="${DT:-0.5}"
TAU="${TAU:-120.0}"
TEMP_BATH="${TEMP_BATH:-300.0}"
SEED="${SEED:-0}"
SIZE_SCALING="${SIZE_SCALING:-0}"
SYNTH_DIST="${SYNTH_DIST:-5.0}"
PRINT_STEP="${PRINT_STEP:-1}"

REACTION_NAME="$(basename "${REACTION_DIR}")"
SCALE_TAG="OFF"
SIZE_FLAG=""
if [[ "${SIZE_SCALING}" == "1" ]]; then
    SCALE_TAG="ON"
    SIZE_FLAG="--size-scaling"
fi

JOB_DATE="$(date +%Y%m%d)"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-synth_${REACTION_NAME}_a${A0}_s${MAX_STEP}_seed${SEED}_${SCALE_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/lhshen/RTIP/research/ic5c02384/results/size-scaling/${OUTPUT_PREFIX}}"

if [[ ! -f "${MODEL}" ]]; then
    echo "DeepMD model not found: ${MODEL}" >&2
    exit 1
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "WORKDIR=${WORKDIR}"
echo "REACTION_DIR=${REACTION_DIR}"
echo "MODEL=${MODEL}"
echo "A0=${A0}"
echo "MAX_STEP=${MAX_STEP}"
echo "DT=${DT}"
echo "TAU=${TAU}"
echo "TEMP_BATH=${TEMP_BATH}"
echo "SEED=${SEED}"
echo "SIZE_SCALING=${SIZE_SCALING}"
echo "SYNTH_DIST=${SYNTH_DIST}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
echo "python=$(command -v python)"
python - <<'PY'
import deepmd, jax
print("deepmd", getattr(deepmd, "__version__", "unknown"))
print("jax", jax.__version__)
print("jax_enable_x64", jax.config.read("jax_enable_x64"))
print("jax_devices", jax.devices())
PY

declare -a RUN_ARGS=(
    --reaction-dir "${REACTION_DIR}"
    --model "${MODEL}"
    --a0 "${A0}"
    --max-step "${MAX_STEP}"
    --dt "${DT}"
    --tau "${TAU}"
    --temp-bath "${TEMP_BATH}"
    --seed "${SEED}"
    --synth-dist "${SYNTH_DIST}"
    --print-step "${PRINT_STEP}"
    --output-dir "${OUTPUT_DIR}"
    --output-prefix "${OUTPUT_PREFIX}"
)
if [[ -n "${SIZE_FLAG}" ]]; then
    RUN_ARGS+=(${SIZE_FLAG})
fi

python examples/ic5c02384/run_size_scaling_synth.py "${RUN_ARGS[@]}"

echo "Outputs:"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.pdb"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.out"
echo "  ${OUTPUT_DIR}/summary.json"
echo "  ${OUTPUT_DIR}/summary.md"
echo "Done."
