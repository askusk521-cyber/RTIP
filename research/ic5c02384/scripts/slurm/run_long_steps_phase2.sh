#!/bin/bash
# Phase 2: Increased steps (2000-5000) for reactions with declining rti_dist
set -euo pipefail

SLURM_SCRIPT="run_rcmd.slurm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export MODE="rc-md"
export TARGET="product"
export TEMP_BATH="300.0"
export INITIAL_TEMP="300.0"
export DT="0.5"
export SEED="0"
export MODEL="/home/lhshen/deepmd_pretrained/DPA-3.2-5M.pt"
export WORKDIR="/home/lhshen/RTIP/rtipmd/jax"

echo "============================================"
echo " Phase 2: Long-steps for declining rti_dist"
echo "============================================"
echo ""

submit_job() {
    local reaction="$1"
    local k="$2"
    local steps="$3"
    
    export REACTION="${reaction}"
    export RC_K="${k}"
    export MAX_STEP="${steps}"

    local k_int=$(awk "BEGIN {printf \"%04d\", ${k}*1000}")
    local temp_int=${TEMP_BATH%.*}
    export OUTPUT_PREFIX="bias_${reaction}_rc-md_product_k${k_int}_T${temp_int}_${steps}"
    export OUTPUT_DIR="/home/lhshen/RTIP/JAX/${OUTPUT_PREFIX}"

    echo "Submitting: ${reaction}  k=${k}  steps=${steps}"
    sbatch --export=ALL --job-name="ls-${reaction:0:10}" --time=48:00:00 "${SCRIPT_DIR}/${SLURM_SCRIPT}"
    echo "  -> ${OUTPUT_DIR}"
    sleep 0.3
}

# 1-tBu__H2CO: rti_dist declining 5.393 -> 2.206 -> 1.178
submit_job "1-tBu__H2CO" "0.01" "2000"
submit_job "1-tBu__H2CO" "0.02" "2000"
submit_job "1-tBu__H2CO" "0.01" "5000"

# 1-tBu__CS2: rti_dist declining 5.144 -> 2.563 -> 1.457
submit_job "1-tBu__CS2" "0.01" "2000"
submit_job "1-tBu__CS2" "0.02" "2000"

# 1-tBu__MeCN: rti_dist declining 5.251 -> 2.759 -> 1.587
submit_job "1-tBu__MeCN" "0.01" "2000"
submit_job "1-tBu__MeCN" "0.02" "2000"

echo ""
echo "Phase 2 done: 7 jobs submitted"
