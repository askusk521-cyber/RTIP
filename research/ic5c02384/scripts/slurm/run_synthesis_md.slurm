#!/bin/bash
#SBATCH -J synth-md
#SBATCH -p main
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

# ============================================================
# RTIP-MD Synthesis: Gaussian-type attractive RTIP + real-PES relaxation
#
# Submit:  sbatch run_synthesis_md.slurm
# Override: SEED=42 MAX_STEP=1000 sbatch run_synthesis_md.slurm
# ============================================================

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

JOB_ID="${SLURM_JOB_ID:-manual}"
MODEL="${MODEL:-/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${JOB_ID}}"

# Reactant XYZ files
SYNTH_DIR="${SYNTH_DIR:-/home/lhshen/RTIP/research/ic5c02384/reactions/1-tBu__CO2}"
SYNTH_DIST="${SYNTH_DIST:-5.0}"
SEED="${SEED:-0}"

# Config file (contains a0, temp_bath, oscillation, etc.)
CONFIG="${CONFIG:-para_synthesis.json}"

mkdir -p "${OUTPUT_DIR}"

# Copy reactants
cp "${SYNTH_DIR}/1.xyz" "${OUTPUT_DIR}/1.xyz"
cp "${SYNTH_DIR}/2.xyz" "${OUTPUT_DIR}/2.xyz"

echo "============================================"
echo "RTIP-MD Synthesis (Gaussian-type attractive)"
echo "============================================"
echo "JOB_ID=${JOB_ID}"
echo "SYNTH_DIR=${SYNTH_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL=${MODEL}"
echo "CONFIG=${CONFIG}"
echo "SEED=${SEED}"
echo "SYNTH_DIST=${SYNTH_DIST}"

cat "${CONFIG}"

python - <<'PY'
import deepmd, jax
print("deepmd", getattr(deepmd, "__version__", "unknown"))
print("jax", jax.__version__)
print("devices", jax.devices())
PY

declare -a CLI_ARGS=(
    --inputs "${OUTPUT_DIR}/1.xyz" "${OUTPUT_DIR}/2.xyz"
    --model "${MODEL}"
    --output-dir "${OUTPUT_DIR}"
    --config "${CONFIG}"
    --seed "${SEED}"
    --synth-dist "${SYNTH_DIST}"
)

# Optional CLI overrides (only add if env var is set)
if [[ -n "${MAX_STEP:-}" ]]; then
    CLI_ARGS+=(--max-step "${MAX_STEP}")
fi
if [[ -n "${OSC_PERIOD:-}" ]]; then
    CLI_ARGS+=(--oscillation-period "${OSC_PERIOD}")
fi
if [[ -n "${OSC_AMP:-}" ]]; then
    CLI_ARGS+=(--oscillation-amplitude "${OSC_AMP}")
fi
if [[ -n "${REDUCTION_RATE:-}" ]]; then
    CLI_ARGS+=(--reduction-rate "${REDUCTION_RATE}")
fi
if [[ -n "${MAX_ROUNDS:-}" ]]; then
    CLI_ARGS+=(--max-rounds "${MAX_ROUNDS}")
fi
if [[ -n "${BOND_TOL:-}" ]]; then
    CLI_ARGS+=(--bond-tolerance "${BOND_TOL}")
fi
if [[ "${NO_PERTURB:-0}" == "1" ]]; then
    CLI_ARGS+=(--no-perturb)
fi
if [[ "${NO_RELAX:-0}" == "1" ]]; then
    CLI_ARGS+=(--no-relax)
fi

echo ""
echo "Running: python -m rtip_jax.cli deepmd-synthesis-md ..."

python -m rtip_jax.cli deepmd-synthesis-md "${CLI_ARGS[@]}"

echo ""
echo "Outputs:"
ls -la "${OUTPUT_DIR}/"
echo "Done."
