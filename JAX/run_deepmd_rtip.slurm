#!/bin/bash
#SBATCH -J rtip-jax-dp
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# Submit from /home/lhshen/RTIP/JAX by default:
#   sbatch run_deepmd_rtip.slurm
#
# Common overrides:
#   INPUT=IS.xyz OUTPUT_DIR=run_pathway MAX_STEP=100 sbatch run_deepmd_rtip.slurm
#   MODE=md INPUT=IS.xyz OUTPUT_DIR=run_md MAX_STEP=1000 sbatch run_deepmd_rtip.slurm
#   METHOD=idwm INPUT=IS.xyz sbatch run_deepmd_rtip.slurm

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-/home/lhshen/RTIP/JAX}}"
cd "${WORKDIR}"

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source /group/software/deepmd-kit-3.1.1/bin/activate
export PYTHONPATH=/home/lhshen/RTIP/JAX/src${PYTHONPATH:+:${PYTHONPATH}}

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-1024}"

MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
INPUT="${INPUT:-IS.xyz}"
OUTPUT_DIR="${OUTPUT_DIR:-run-${SLURM_JOB_ID:-manual}}"
MODE="${MODE:-pathway}"       # pathway or md
METHOD="${METHOD:-rtip}"      # rtip or idwm, only for MODE=pathway
SEED="${SEED:-0}"
CONFIG="${CONFIG:-}"
MAX_STEP="${MAX_STEP:-}"

if [[ ! -f "${MODEL}" ]]; then
    echo "DeepMD model not found: ${MODEL}" >&2
    exit 1
fi

if [[ ! -f "${INPUT}" ]]; then
    echo "Input XYZ not found: ${INPUT}" >&2
    echo "Set INPUT=/path/to/file.xyz when submitting." >&2
    exit 1
fi

declare -a COMMON_ARGS=(
    --input "${INPUT}"
    --model "${MODEL}"
    --output-dir "${OUTPUT_DIR}"
    --seed "${SEED}"
)

if [[ -n "${CONFIG}" ]]; then
    COMMON_ARGS+=(--config "${CONFIG}")
fi

if [[ -n "${MAX_STEP}" ]]; then
    COMMON_ARGS+=(--max-step "${MAX_STEP}")
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "WORKDIR=${WORKDIR}"
echo "MODEL=${MODEL}"
echo "INPUT=${INPUT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODE=${MODE}"
echo "METHOD=${METHOD}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "python=$(command -v python)"
python - <<'PY'
import deepmd
import jax
print("deepmd", getattr(deepmd, "__version__", "unknown"))
print("jax", jax.__version__)
print("jax_enable_x64", jax.config.read("jax_enable_x64"))
print("jax_devices", jax.devices())
PY

case "${MODE}" in
    pathway)
        python -m rtip_jax.cli deepmd-pathway \
            "${COMMON_ARGS[@]}" \
            --method "${METHOD}"
        ;;
    md)
        declare -a MD_ARGS=("${COMMON_ARGS[@]}")
        if [[ "${NO_PERTURB:-0}" == "1" ]]; then
            MD_ARGS+=(--no-perturb)
        fi
        python -m rtip_jax.cli deepmd-md "${MD_ARGS[@]}"
        ;;
    *)
        echo "Unknown MODE=${MODE}; expected 'pathway' or 'md'." >&2
        exit 1
        ;;
esac

echo "Done."
