#!/bin/bash
#SBATCH --job-name=r3-8-p2b
#SBATCH --exclude=node002
#SBATCH --nice=2000
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --array=0-31%4
#SBATCH --output=logs/r3_8_p2b_%A_%a.out
# R3 item 8 / P2-b (CPU): one (dataset, arch, seed, fold) cell per task, list in $LIST.
set -euo pipefail
R=$HOME/DST-Skin
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
LIST=${LIST:-$HOME/r3work/item8/p2b_list.txt}
read ds arch seed fold <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")"
cd $R
[ -s results/r3/8/p2b/cells/${ds}_${arch}_s${seed}_${fold}.json ] && { echo done; exit 0; }
$HOME/.venvs/crossfit-r3/bin/python scripts/r3/item8_p2b.py --ds $ds --arch $arch --seed $seed --fold $fold
