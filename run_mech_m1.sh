#!/bin/bash
#SBATCH --job-name=mech-m1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/mech_m1_%A_%a.out
# DST_PY=... M1_CELLS="ds:model:arch:variant[:fold] ..." sbatch --array=0-(n-1)%k run_mech_m1.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=${DST_PY:-python}
read -ra CELLS <<< "${M1_CELLS}"
IFS=: read -r DS MODEL ARCH VAR FOLD <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
case ${DS} in camelyon) export DST_STAGE_DS=camelyon17_v1.0 ;; iwildcam) export DST_STAGE_DS=iwildcam_v2.0 ;; rxrx1) export DST_STAGE_DS=rxrx1_v1.0 ;; esac
export DST_STAGE_DATA=1
source scripts/rigor/stage_wilds.sh
stage_run ${PY} scripts/rigor/mech_m1_extract.py --ds "${DS}" --model "${MODEL}" --arch "${ARCH}" --variant "${VAR}" --fold "${FOLD:-0}"
