#!/bin/bash
#SBATCH --job-name=r3-5b-B
#SBATCH --exclude=node002
#SBATCH --nice=10000
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH --array=0-9%5
#SBATCH --output=logs/r3_5b_B_%A_%a.out
# R3 item 5b Step B point estimates (CPU): task = FM x scorer. JK=1 adds the jackknife CI.
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export R3_INPUTS=${R3_INPUTS:-$HOME/r3work/item1/inputs}
cd $R
FMS=(uni virchow2 dinov2_vitb14 dinov2_vitl14 conch_v1_5)
SCS=(mahalanobis_l2 knn_mean_cosine)
T=$SLURM_ARRAY_TASK_ID
FM=${FMS[$((T % 5))]}
SC=${SCS[$((T / 5))]}
$PY scripts/r3/item5b_stepB.py --fm $FM --scorer $SC ${JK:+--jackknife} ${KS:+--ks $KS}
