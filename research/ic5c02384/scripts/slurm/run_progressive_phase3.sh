#!/bin/bash
# Phase 3: Progressive restraint for high-temp reactions
#   Start with low initial temp, progressively tighten k
# Targets: 1-Me__CO2, 1-H__CO2, 1-CN__CO2
# Run from: /home/lhshen/RTIP/research/ic5c02384/scripts/slurm/
set -euo pipefail

SLURM_SCRIPT="run_progressive.slurm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export TARGET="product"
export K_STAGES="0.006,0.01,0.02"
export STEPS_PER_STAGE="500"
export TEMP_BATH="300.0"
export INITIAL_TEMP="150.0"
export SEED="0"
export MODEL="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt"
export WORKDIR="/home/lhshen/RTIP/rtipmd/jax"

# High-temp runaway reactions (small substituents)
REACTIONS=(
    "1-Me__CO2"
    "1-H__CO2"
    "1-CN__CO2"
)

echo "============================================"
echo " Phase 3: Progressive restraint"
echo " k stages: ${K_STAGES}"
echo " steps/stage: ${STEPS_PER_STAGE}"
echo " initial temp: ${INITIAL_TEMP} K"
echo "============================================"
echo ""

for reaction in "${REACTIONS[@]}"; do
    export REACTION="${reaction}"
    export OUTPUT_PREFIX="prog_${reaction}_product_3stage_k006_001_002"
    export OUTPUT_DIR="/home/lhshen/RTIP/JAX/${OUTPUT_PREFIX}"

    echo "Submitting: REACTION=${reaction}"
    sbatch --export=ALL --job-name="prog-${reaction:0:10}" "${SCRIPT_DIR}/${SLURM_SCRIPT}"
    echo "  ${OUTPUT_DIR}"
    sleep 0.3
done

echo ""
echo "Phase 3 done: 3 jobs submitted"
