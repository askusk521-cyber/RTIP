#!/bin/bash
# Batch submit RC-MD jobs for remaining ic5c02384 reactions.
#
# Usage:
#   cd /home/lhshen/RTIP/rtipmd/jax
#   bash run_rcmd_batch.sh
#
# All jobs use: rc-md, product target, k=0.006, 1000 steps, 300 K, DPA-3.2-5M

set -euo pipefail

SLURM_SCRIPT="run_rcmd.slurm"

export MODE="rc-md"
export TARGET="product"
export RC_K="0.006"
export MAX_STEP="1000"
export TEMP_BATH="300.0"
export INITIAL_TEMP="300.0"
export DT="0.5"
export SEED="0"
export MODEL="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt"

echo "============================================"
echo " RC-MD Batch: ic5c02384 remaining reactions"
echo " mode=${MODE}  target=${TARGET}  rc_k=${RC_K}"
echo " max_step=${MAX_STEP}  temp=${TEMP_BATH}K"
echo "============================================"
echo ""

# ---- Batch A: CO2 substituent variants (high priority) ----
echo "=== Batch A: CO2 substituent variants ==="
echo ""

CO2_REACTIONS=(
    "1-Me__CO2"
    "1-PMe2__CO2"
    "1-Ph__CO2"
    "1-SiMe3__CO2"
)

for reaction in "${CO2_REACTIONS[@]}"; do
    export REACTION="${reaction}"
    echo "Submitting: REACTION=${REACTION}"
    sbatch --export=ALL --job-name="rc-${reaction:0:12}" "${SLURM_SCRIPT}"
    echo "  submitted."
    echo ""
    sleep 0.5
done

# ---- Batch B: Small molecule scope (medium priority) ----
echo "=== Batch B: Small molecule scope ==="
echo ""

SMALL_MOL_REACTIONS=(
    "1-tBu__H2CO"
    "1-tBu__CS2"
    "1-tBu__MeCN"
    "1-tBu__MeCH=NMe"
)

for reaction in "${SMALL_MOL_REACTIONS[@]}"; do
    export REACTION="${reaction}"
    echo "Submitting: REACTION=${REACTION}"
    sbatch --export=ALL --job-name="rc-${reaction:0:12}" "${SLURM_SCRIPT}"
    echo "  submitted."
    echo ""
    sleep 0.5
done

# ---- Batch C: Product-only (low priority) ----
echo "=== Batch C: Product-only (no TS reference) ==="
echo ""

PRODUCT_ONLY_REACTIONS=(
    "1-H__CO2"
    "1-CN__CO2"
)

for reaction in "${PRODUCT_ONLY_REACTIONS[@]}"; do
    export REACTION="${reaction}"
    echo "Submitting: REACTION=${REACTION}"
    sbatch --export=ALL --job-name="rc-${reaction:0:12}" "${SLURM_SCRIPT}"
    echo "  submitted."
    echo ""
    sleep 0.5
done

echo "============================================"
echo " All 10 jobs submitted."
echo " Monitor: squeue -u \$USER"
echo "============================================"
