#!/bin/bash
#SBATCH --job-name=t1_G1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --exclude=node002
#SBATCH --requeue
#SBATCH --time=06:00:00
#SBATCH --array=0-4%1
#SBATCH --output=/data2/cmdir/home/toandq/DST-Skin/logs/t1/G1_%A_%a.out
# T1 item G1 (frozen backbones): task i -> FM i of (uni, virchow2, dinov2_vitb14, dinov2_vitl14, conch_v1_5).
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
FMS=(uni virchow2 dinov2_vitb14 dinov2_vitl14 conch_v1_5)
cd $HOME/DST-Skin
$HOME/.venvs/crossfit-r3/bin/python scripts/t1/item_G.py g1 ${FMS[$SLURM_ARRAY_TASK_ID]}
echo DONE
