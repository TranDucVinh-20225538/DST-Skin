#!/bin/bash
#SBATCH --job-name=r3-8-p2d-smoke
#SBATCH --exclude=node002
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/r3_8_p2d_smoke_%j.out
# R3 item 8 / P2-d technical smoke (no results used): one CNN run (kvasir seg resnet50_s42, 1 epoch, 512 images), both FMs on
# 2,000 Kvasir images, scoring of those runs with 3 jackknife blocks; then the smoke outputs are deleted. Fails -> chain stops.
set -euo pipefail
cd $HOME/DST-Skin
S=scripts/r3/run_item8_p2d.sh
export SMOKE=1
MODE=train SLURM_ARRAY_TASK_ID=4 bash $S
for t in 0 1; do MODE=fm SLURM_ARRAY_TASK_ID=$t bash $S; done
for t in 12 13 20 21 22 23; do MODE=score SLURM_ARRAY_TASK_ID=$t bash $S; done
ls results/r3/8/p2d/cells/*_smoke* results/r3/8/p2d/train/*_smoke*
rm -f results/r3/8/p2d/cells/*_smoke* results/r3/8/p2d/train/*_smoke* outputs/rigor_pack/r3_item8/p2d/*_smoke* data/models/r3_item8/p2d/*_smoke*
echo SMOKE_OK
