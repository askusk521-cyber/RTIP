#!/bin/bash
# Phase 1: k=0.01, 0.02, 0.05 scan for 10 failed reactions
# Run from: /home/lhshen/RTIP/research/ic5c02384/scripts/slurm/
set -euo pipefail

SLURM_SCRIPT="run_rcmd.slurm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MODE="rc-md"
export TARGET="product"
export MAX_STEP="1000"
export TEMP_BATH="300.0"
export INITIAL_TEMP="300.0"
export DT="0.5"
export SEED="0"
export MODEL="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt"
export WORKDIR="/home/lhshen/RTIP/rtipmd/jax"

K_VALUES=("0.01" "0.02" "0.05")

REACTIONS=(
    "1-Me__CO2"
    "1-PMe2__CO2"
    "1-Ph__CO2"
    "1-SiMe3__CO2"
    "1-tBu__H2CO"
    "1-tBu__CS2"
    "1-tBu__MeCN"
    "1-tBu__MeCH=NMe"
    "1-H__CO2"
    "1-CN__CO2"
)

echo "============================================"
echo " Phase 1: k-scan for 10 failed reactions"
echo " k values: ${K_VALUES[*]}"
echo " Total jobs: $(( ${#K_VALUES[@]} * ${#REACTIONS[@]} ))"
echo "============================================"
echo ""

job_count=0
for reaction in "${REACTIONS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        export REACTION="${reaction}"
        export RC_K="${k}"

        k_int=$(awk "BEGIN {printf \"%04d\", ${k}*1000}")
        temp_int=${TEMP_BATH%.*}
        export OUTPUT_PREFIX="bias_${reaction}_rc-md_product_k${k_int}_T${temp_int}_${MAX_STEP}"
        export OUTPUT_DIR="/home/lhshen/RTIP/JAX/${OUTPUT_PREFIX}"

        echo "Submitting: REACTION=${reaction}  RC_K=${k}"
        sbatch --export=ALL --job-name="p1-${reaction:0:10}" "${SCRIPT_DIR}/${SLURM_SCRIPT}"
        echo "  ${OUTPUT_DIR}"
        job_count=$((job_count + 1))
        sleep 0.3
    done
done

echo ""
echo "Phase 1 done: ${job_count} jobs submitted"
