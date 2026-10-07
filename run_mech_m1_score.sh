#!/bin/bash
#SBATCH --job-name=mech-m1-score
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/mech_m1_score_%A_%a.out
# M1S_CELLS="ds:arch:variant ..." sbatch --array=0-(n-1)%k run_mech_m1_score.sh
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
read -ra CELLS <<< "${M1S_CELLS}"
IFS=: read -r DS ARCH VAR <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
${PY} scripts/rigor/mech_m1_score.py --ds "${DS}" --arch "${ARCH}" --variant "${VAR}"
