#!/bin/bash
# Submit 10 identical RepulsivePot RTIPMD runs with different seeds.
# Gentle conditions: low a0, low temperature, many steps.
#
# Usage (on n5):
#   cd /home/lhshen/RTIP/rtipmd/jax
#   bash ../../research/ic5c02384/scripts/slurm/run_gentle_rtipmd_batch.sh
#
# Config:  /home/lhshen/RTIP/research/ic5c02384/para_gentle.json
# Results: /home/lhshen/RTIP/research/ic5c02384/results/rtipmd-gentle/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- fixed parameters (n5 paths) ----
SLURM_SCRIPT="${SCRIPT_DIR}/run_deepmd_rtip.slurm"
CONFIG="/home/lhshen/RTIP/research/ic5c02384/para_gentle.json"
REACTION_DIR="/home/lhshen/RTIP/research/ic5c02384/reactions/1-tBu__CO2"
RESULTS_DIR="/home/lhshen/RTIP/research/ic5c02384/results/rtipmd-gentle"

export MODE="md"
export SYNTH_INPUTS="${REACTION_DIR}/1.xyz ${REACTION_DIR}/2.xyz"
export SYNTH_DIST="5.0"
export CONFIG="${CONFIG}"
# MAX_STEP is set in the config file, don't override here
export MAX_STEP=""

echo "============================================"
echo " Gentle RTIPMD Batch Submission"
echo "============================================"
echo " Reaction:     1-tBu__CO2"
echo " Config:       ${CONFIG}"
echo " Results base: ${RESULTS_DIR}"
echo " Seeds:        0..9 (10 runs)"
echo " a0:           0.0005"
echo " temp_bath:    300 K"
echo " max_step:     5000"
echo "============================================"
echo ""

for seed in $(seq 0 9); do
    RUN_NAME="seed${seed}_a00005_T300_5000"
    export SEED="${seed}"
    export OUTPUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
    export OUTPUT_PREFIX="rtipmd_1tBu_CO2_${RUN_NAME}"

    echo "Submitting seed=${seed}  ->  ${OUTPUT_DIR}"

    sbatch --export=ALL \
           --job-name="gmd-${seed}" \
           "${SLURM_SCRIPT}"

    echo "  submitted."
    sleep 0.5
done

echo ""
echo "Done. Check job status with: squeue -u \$USER"
echo "Results will be in: ${RESULTS_DIR}"
