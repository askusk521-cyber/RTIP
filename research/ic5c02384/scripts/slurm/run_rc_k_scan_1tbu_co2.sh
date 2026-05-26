#!/bin/bash
# Batch submit RC-MD k-scan jobs for 1-tBu + CO2 coupling verification.
#
# Usage:
#   cd /home/lhshen/RTIP/rtipmd/jax
#   bash run_rc_k_scan_1tbu_co2.sh
#
# Each job: rc-md, 1000 steps, 300 K, DeePMD DPA-3.2-5M, product or TS target.

set -euo pipefail

SLURM_SCRIPT="run_bias_exploration_1tbu_co2.slurm"

# ---- fixed parameters ----
export MODE="rc-md"
export MAX_STEP="1000"
export TEMP_BATH="300.0"
export INITIAL_TEMP="300.0"
export DT="0.5"
export SEED="0"
export MODEL="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt"
export INPUT="examples/ic5c02384/reactions/1-tBu__CO2/IS.xyz"
export REACTION_DIR="examples/ic5c02384/reactions/1-tBu__CO2"

# ---- RC_K values to scan (product target, the promising one) ----
RC_K_VALUES=(
  0.001
  0.002
  0.004
  0.006
  0.008
  0.010
)

echo "============================================"
echo " RC-MD k-scan: 1-tBu + CO2 coupling"
echo " target = product"
echo " max_step = ${MAX_STEP}"
echo " k values = ${RC_K_VALUES[*]}"
echo "============================================"
echo ""

for k in "${RC_K_VALUES[@]}"; do
    export TARGET="product"
    export RC_K="${k}"
    k_int=$(awk "BEGIN {printf \"%04d\", ${k}*1000}")
    export OUTPUT_PREFIX="bias_1tBu_CO2_rc-md_product_k${k_int}_T300_${MAX_STEP}"

    echo "Submitting: RC_K=${k}  OUTPUT_PREFIX=${OUTPUT_PREFIX}"

    sbatch --export=ALL \
           --job-name="rc-k${k}-p" \
           "${SLURM_SCRIPT}"

    echo "  submitted."
    echo ""

    # Small pause to avoid hammering the scheduler
    sleep 0.5
done

# ---- also scan a few TS-target jobs for comparison ----
echo "============================================"
echo " TS-target comparison jobs"
echo "============================================"
echo ""

TS_K_VALUES=(0.002 0.006 0.010)

for k in "${TS_K_VALUES[@]}"; do
    export TARGET="ts"
    export RC_K="${k}"
    k_int=$(awk "BEGIN {printf \"%04d\", ${k}*1000}")
    export OUTPUT_PREFIX="bias_1tBu_CO2_rc-md_ts_k${k_int}_T300_${MAX_STEP}"

    echo "Submitting: RC_K=${k}  OUTPUT_PREFIX=${OUTPUT_PREFIX}"

    sbatch --export=ALL \
           --job-name="rc-k${k}-t" \
           "${SLURM_SCRIPT}"

    echo "  submitted."
    echo ""

    sleep 0.5
done

echo "Done. Use 'squeue -u \$USER' to monitor."
