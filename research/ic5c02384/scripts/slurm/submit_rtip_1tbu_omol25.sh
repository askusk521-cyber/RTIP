#!/bin/bash
# Batch submit 1-tBu__CO2 RTIP Gaussian + OMol25 jobs
set -euo pipefail

SLURM_SCRIPT="../../research/ic5c02384/scripts/slurm/run_rtip_attractive_1tbu_co2.slurm"
MODEL="/home/lhshen/deepmd_pretrained/frozen_OMol25.pth"

# Fixed params
export MODE="attractive-md"
export TARGET="product"
export MODEL="${MODEL}"
export ZERO_VELOCITY="1"
export DT="0.5"
export SEED="0"

echo "=== Submitting RTIP Gaussian + OMol25 jobs for 1-tBu__CO2 ==="
echo "Model: ${MODEL}"
echo ""

job_count=0

# Job 1: a0=0.005, 1000 steps, 300K
export A0="0.005"
export MAX_STEP="1000"
export TEMP_BATH="300.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.005_T300_s1000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.005 step=1000 T=300  => ${job_id}"

# Job 2: a0=0.01, 1000 steps, 300K
export A0="0.01"
export MAX_STEP="1000"
export TEMP_BATH="300.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.01_T300_s1000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.01  step=1000 T=300  => ${job_id}"

# Job 3: a0=0.02, 1000 steps, 300K
export A0="0.02"
export MAX_STEP="1000"
export TEMP_BATH="300.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.02_T300_s1000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.02  step=1000 T=300  => ${job_id}"

# Job 4: a0=0.005, 2000 steps, 300K
export A0="0.005"
export MAX_STEP="2000"
export TEMP_BATH="300.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.005_T300_s2000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.005 step=2000 T=300  => ${job_id}"

# Job 5: a0=0.01, 2000 steps, 300K
export A0="0.01"
export MAX_STEP="2000"
export TEMP_BATH="300.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.01_T300_s2000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.01  step=2000 T=300  => ${job_id}"

# Job 6: a0=0.02, 5000 steps, 300K
export A0="0.02"
export MAX_STEP="5000"
export TEMP_BATH="300.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.02_T300_s5000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.02  step=5000 T=300  => ${job_id}"

# Job 7: a0=0.01, 2000 steps, 600K
export A0="0.01"
export MAX_STEP="2000"
export TEMP_BATH="600.0"
export OUTPUT_PREFIX="bias_1tBu_CO2_attractive_product_a0.01_T600_s2000_zeroV"
job_id=$(sbatch --parsable "${SLURM_SCRIPT}" 2>&1)
echo "[$((++job_count))/7] a0=0.01  step=2000 T=600  => ${job_id}"

echo ""
echo "=== All 7 jobs submitted ==="
