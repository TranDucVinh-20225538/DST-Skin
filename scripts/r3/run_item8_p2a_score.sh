#!/bin/bash
#SBATCH --job-name=r3-8-p2a-score
#SBATCH --exclude=node002
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r3_8_p2a_score_%A_%a.out
# R3 item 8 / P2-a scoring (CPU). MODE=score: array 0-11 = 6 runs x {mahalanobis_l2, knn_mean_cosine}
# (skips cells already written); MODE=probe: array 0-1 = dinov2_vitb14, rad_dino linear probes. NB=<n> for a smoke run.
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
export PYTHONPATH=$R OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $R
RUNS=(resnet50_s42 resnet50_s43 convnext_tiny_s42 convnext_tiny_s43 dinov2_vitb14 rad_dino)
if [ "$MODE" = probe ]; then
  FMS=(dinov2_vitb14 rad_dino)
  $PY scripts/r3/item8_p2a_probe.py ${FMS[$SLURM_ARRAY_TASK_ID]}
  exit 0
fi
SC=(mahalanobis_l2 knn_mean_cosine)
run=${RUNS[$((SLURM_ARRAY_TASK_ID / 2))]}; sc=${SC[$((SLURM_ARRAY_TASK_ID % 2))]}
X=(); [ -n "${NB:-}" ] && X=(--nb $NB)
[ -z "${NB:-}" ] && [ -s results/r3/8/p2a/cells/${run}_${sc}.json ] && { echo done; exit 0; }
$PY scripts/r3/item8_p2a_score.py --run $run --scorer $sc "${X[@]}"
