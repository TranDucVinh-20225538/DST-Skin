#!/bin/bash
#SBATCH --job-name=mech-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/mech_cpu_%A_%a.out
# Cells: MECH_CELLS="ds:arch[:m4]" list; array index picks one.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python
read -ra CELLS <<< "${MECH_CELLS}"
IFS=: read -r DS ARCH M4 <<< "${CELLS[${SLURM_ARRAY_TASK_ID}]}"
${PY} scripts/rigor/mech_cpu.py --ds "${DS}" --arch "${ARCH}" ${M4:+--m4-logit}
