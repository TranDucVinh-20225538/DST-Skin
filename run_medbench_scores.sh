#!/bin/bash
#SBATCH --job-name=med-score
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --exclude=node002
#SBATCH --output=logs/medbench_score_%A_%a.out
# MS_RUNS="ds:arch:seed:arm ..." sbatch --array=0-(k-1) run_medbench_scores.sh   (task i scores runs i, i+k, ...)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset PYTHONPATH
export PYTHONPATH=$(pwd) OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
PY=${DST_PY:-python}
read -ra RUNS <<< "${MS_RUNS}"
K=${SLURM_ARRAY_TASK_COUNT:-1}
for ((i=${SLURM_ARRAY_TASK_ID:-0}; i<${#RUNS[@]}; i+=K)); do
    IFS=: read -r DS ARCH SEED ARM <<< "${RUNS[$i]}"
    OUT="outputs/reports/rigor_pack/medbench/${DS}/scores_${ARCH}_s${SEED}_${ARM}.json"
    [ -s "${OUT}" ] && continue
    if [ ! -s "outputs/rigor_pack/medbench/${DS}/${ARCH}_s${SEED}_${ARM}.npz" ]; then
        echo "MISSING npz ${DS} ${ARCH} s${SEED} ${ARM}"; continue
    fi
    t0=$(date +%s)
    if ${PY} scripts/rigor/medbench_scores.py --ds "${DS}" --arch "${ARCH}" --seed "${SEED}" --arm "${ARM}" > /dev/null; then
        echo "${DS} ${ARCH} s${SEED} ${ARM} $(( $(date +%s) - t0 ))s"
    else
        echo "FAIL ${DS} ${ARCH} s${SEED} ${ARM}"
    fi
done
