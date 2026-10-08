#!/bin/bash
#SBATCH --job-name=r3-8-p2a
#SBATCH --exclude=node002
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/r3_8_p2a_%x_%A_%a.out
# R3 item 8 / P2-a (results/r3/8/PRECOMMIT.json), M_std = wang2017 (Phase 0).
# MODE=prep / prep_ood : sets / OOD staging (CPU; submit with --gres=none -c 4 --mem 32G)
# MODE=cnn  : array 0-3 = resnet50 s42, resnet50 s43, convnext_tiny s42, convnext_tiny s43 (staged images in node-local /tmp)
# MODE=cnn_ood : same array, OOD features from the saved best checkpoints
# MODE=fm   : array 0-1 = dinov2_vitb14, rad_dino (archives streamed, no node-local copy)
# SMOKE=1   : 1 epoch / 512 images per set, outputs tagged _smoke
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
cd $R
export PYTHONPATH=$R OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
case "$MODE" in
prep)
  $PY scripts/r3/item8_p2a_prep.py ;;
prep_ood)
  $PY scripts/r3/item8_p2a_prep.py --ood ;;
cnn|cnn_ood)
  RUNS=(resnet50:42 resnet50:43 convnext_tiny:42 convnext_tiny:43)
  IFS=: read -r ARCH SEED <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"
  T=$(mktemp -d /tmp/r3p2a_XXXXXX)
  trap 'rm -rf "$T"' EXIT
  mkdir -p $T/img $T/ood
  t0=$(date +%s)
  X=(); [ "${SMOKE:-0}" = 1 ] && X=(--max-epochs 1 --max-n 512)
  if [ "$MODE" = cnn ]; then
    for f in data/staged/nih_256/images_*.tar; do tar -xf $f -C $T/img; done
  else
    for f in data/staged/p2a_ood_256/*.tar; do tar -xf $f -C $T/ood; done
    X+=(--ood-only)
  fi
  echo "staged $(ls $T/img | wc -l) NIH + $(ls $T/ood | wc -l) OOD in $(( $(date +%s) - t0 ))s on $(hostname)"
  $PY scripts/r3/item8_p2a_train.py --arch $ARCH --seed $SEED --img-dir $T/img --ood-dir $T/ood --workers 10 "${X[@]}" ;;
fm)
  FMS=(dinov2_vitb14 rad_dino)
  FM=${FMS[$SLURM_ARRAY_TASK_ID]}
  [ "$FM" = rad_dino ] && export PYTHONPATH=$HOME/r3work/pylibs/hf444:$PYTHONPATH
  X=(); [ "${SMOKE:-0}" = 1 ] && X=(--max-n 512)
  $PY scripts/r3/item8_p2a_fm_extract.py --fm $FM --workers 12 "${X[@]}" ;;
*) echo "MODE?"; exit 2 ;;
esac
