#!/bin/bash
#SBATCH --job-name=t1_D_cma
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=75G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/D_cma_%j.out
# T1 item D2: CMA-ES (population 32, 40 generations, seed 401) after the 3,000 random configs.
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $HOME/DST-Skin
[ -f results/t1/D/raw/d2_cma.done ] && exit 0
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_D_sim.py cma
echo DONE
