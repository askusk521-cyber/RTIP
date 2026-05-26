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

# Submit from /home/lhshen/RTIP/rtipmd/jax by default:
#   sbatch run_deepmd_rtip.slurm
#
# Common overrides:
#   MAX_STEP=100 sbatch run_deepmd_rtip.slurm
#   SYNTH_INPUTS="1.xyz 2.xyz 3.xyz" MAX_STEP=100 sbatch run_deepmd_rtip.slurm
#   MODE=md MAX_STEP=1000 sbatch run_deepmd_rtip.slurm
#   METHOD=idwm sbatch run_deepmd_rtip.slurm
#   SYNTHESIZE=0 INPUT=IS.xyz sbatch run_deepmd_rtip.slurm

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-/home/lhshen/RTIP/rtipmd/jax}}"
cd "${WORKDIR}"

export CUDA_HOME=/group/software/cuda-12.9.1
export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Conda activation scripts may read unset toolchain variables such as ADDR2LINE.
set +u
source /group/software/deepmd-kit-3.1.1/bin/activate
set -u
export PYTHONPATH=/home/lhshen/RTIP/rtipmd/jax/src${PYTHONPATH:+:${PYTHONPATH}}

export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-${OMP_NUM_THREADS}}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export DP_INFER_BATCH_SIZE="${DP_INFER_BATCH_SIZE:-1024}"

if [[ -n "${INPUT+x}" ]]; then
    INPUT_WAS_SET=1
else
    INPUT_WAS_SET=0
fi
if [[ -n "${SYNTH_INPUTS+x}" ]]; then
    SYNTH_INPUTS_WAS_SET=1
else
    SYNTH_INPUTS_WAS_SET=0
fi

JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${JOB_ID}}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${JOB_ID}}"
if [[ "${INPUT_WAS_SET}" -eq 1 ]]; then
    DEFAULT_SYNTHESIZE=0
else
    DEFAULT_SYNTHESIZE=1
fi
SYNTHESIZE="${SYNTHESIZE:-${DEFAULT_SYNTHESIZE}}"
if [[ "${SYNTH_INPUTS_WAS_SET}" -eq 1 ]]; then
    SYNTH_INPUTS="${SYNTH_INPUTS}"
elif [[ -f 1.xyz && -f 2.xyz ]]; then
    SYNTH_INPUTS="1.xyz 2.xyz"
else
    shopt -s nullglob
    TOP_LEVEL_XYZ=( *.xyz )
    shopt -u nullglob
    if [[ "${#TOP_LEVEL_XYZ[@]}" -ge 2 && "${#TOP_LEVEL_XYZ[@]}" -le 4 ]]; then
        SYNTH_INPUTS="${TOP_LEVEL_XYZ[*]}"
    else
        SYNTH_INPUTS="1.xyz 2.xyz"
    fi
fi
SYNTH_DIST="${SYNTH_DIST:-5.0}"
INPUT="${INPUT:-${OUTPUT_DIR}/${OUTPUT_PREFIX}-IS.xyz}"
MODE="${MODE:-pathway}"       # pathway or md
METHOD="${METHOD:-rtip}"      # rtip or idwm, only for MODE=pathway
SEED="${SEED:-0}"
CONFIG="${CONFIG:-}"
MAX_STEP="${MAX_STEP:-}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${MODEL}" ]]; then
    echo "DeepMD model not found: ${MODEL}" >&2
    exit 1
fi

if [[ "${SYNTHESIZE}" != "0" ]]; then
    read -r -a SYNTH_INPUT_ARRAY <<< "${SYNTH_INPUTS}"
    if [[ "${#SYNTH_INPUT_ARRAY[@]}" -eq 0 ]]; then
        echo "SYNTH_INPUTS is empty; provide 2, 3, or 4 XYZ files, or set SYNTHESIZE=0 with INPUT=..." >&2
        exit 1
    fi
    for xyz in "${SYNTH_INPUT_ARRAY[@]}"; do
        if [[ ! -f "${xyz}" ]]; then
            echo "Synthesis input XYZ not found: ${xyz}" >&2
            exit 1
        fi
    done
    echo "Generating synthesis initial state: ${INPUT}"
    python -m rtip_jax.cli synthesize \
        --inputs "${SYNTH_INPUT_ARRAY[@]}" \
        --output "${INPUT}" \
        --dist "${SYNTH_DIST}" \
        --seed "${SEED}"
else
    if [[ ! -f "${INPUT}" ]]; then
        echo "Input XYZ not found: ${INPUT}" >&2
        echo "Set INPUT=/path/to/file.xyz, or leave SYNTHESIZE=1 to generate it from SYNTH_INPUTS." >&2
        exit 1
    fi
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
echo "JOB_ID=${JOB_ID}"
echo "WORKDIR=${WORKDIR}"
echo "MODEL=${MODEL}"
echo "SYNTHESIZE=${SYNTHESIZE}"
echo "SYNTH_INPUTS=${SYNTH_INPUTS}"
echo "SYNTH_DIST=${SYNTH_DIST}"
echo "INPUT=${INPUT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_PREFIX=${OUTPUT_PREFIX}"
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

if [[ -f "${OUTPUT_DIR}/rtip.pdb" ]]; then
    mv -f "${OUTPUT_DIR}/rtip.pdb" "${OUTPUT_DIR}/${OUTPUT_PREFIX}.pdb"
fi
if [[ -f "${OUTPUT_DIR}/rtip.out" ]]; then
    mv -f "${OUTPUT_DIR}/rtip.out" "${OUTPUT_DIR}/${OUTPUT_PREFIX}.out"
fi

echo "Outputs:"
echo "  ${INPUT}"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.pdb"
echo "  ${OUTPUT_DIR}/${OUTPUT_PREFIX}.out"
echo "Done."
