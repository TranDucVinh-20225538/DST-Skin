#!/bin/bash
#SBATCH --job-name=r3-8-p2d
#SBATCH --exclude=node002
#SBATCH --output=logs/r3_8_p2d_%x_%A_%a.out
# R3 item 8 / P2-d (results/r3/8/p2d/PRECOMMIT.json); primary candidates from results/r3/8/p2d/pilot_ood.json.
# MODE=train : GPU, array 0-11 = (kvasir lit, kvasir seg, brain rec) x (resnet50 s42, s43, convnext_tiny s42, s43);
#              staged PNGs copied to node-local /tmp, removed on exit
# MODE=fm    : GPU, array 0-3 = (kvasir, brain) x (dinov2_vitb14, biomedclip)
# MODE=score : CPU, array 0-35 = (kvasir lit, kvasir seg, brain rec) x 6 runs x 2 scorers (skips cells already written)
# MODE=report: CPU, writes results/r3/8/p2d/REPORT.md
# SMOKE=1    : train 1 epoch / 512 images (task 4 only), fm 2,000 images, score 3 blocks / 20 bootstrap; outputs tagged _smoke
set -euo pipefail
R=$HOME/DST-Skin
PY=$HOME/.venvs/crossfit-r3/bin/python
cd $R
export PYTHONPATH=$R HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
N=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N
DV=(kvasir:lit kvasir:seg brain:rec)
RUNS=(resnet50_s42 resnet50_s43 convnext_tiny_s42 convnext_tiny_s43 dinov2_vitb14 biomedclip)
T=${SLURM_ARRAY_TASK_ID:-0}
case "$MODE" in
train)
  IFS=: read -r DS VAR <<< "${DV[$((T / 4))]}"
  RUN=${RUNS[$((T % 4))]}; ARCH=${RUN%_s*}; SEED=${RUN##*_s}
  X=(); [ "${SMOKE:-0}" = 1 ] && X=(--max-epochs 1 --max-n 512)
  TAG=${DS}_${VAR}_${RUN}; [ "${SMOKE:-0}" = 1 ] && TAG=${TAG}_smoke
  [ -s results/r3/8/p2d/train/$TAG.json ] && { echo "done $TAG"; exit 0; }
  D=$(mktemp -d /tmp/r3p2d_XXXXXX); trap 'rm -rf "$D"' EXIT
  t0=$(date +%s)
  for f in data/staged/p2d_${DS}_256/*.tar; do tar -xf $f -C $D; done
  echo "staged $(ls $D | wc -l) images in $(( $(date +%s) - t0 ))s on $(hostname)"
  $PY scripts/r3/item8_p2d_train.py --ds $DS --variant $VAR --arch $ARCH --seed $SEED --img-dir $D --workers $((N - 1)) "${X[@]}" ;;
fm)
  DSS=(kvasir kvasir brain brain); FM=(dinov2_vitb14 biomedclip dinov2_vitb14 biomedclip)
  X=(); [ "${SMOKE:-0}" = 1 ] && X=(--max-n 2000)
  P=$PY; [ "${FM[$T]}" = biomedclip ] && P=$HOME/.venvs/biomedclip/bin/python
  $P scripts/r3/item8_p2d_fm_extract.py --ds ${DSS[$T]} --fm ${FM[$T]} "${X[@]}" ;;
score)
  SC=(mahalanobis_l2 knn_mean_cosine)
  IFS=: read -r DS VAR <<< "${DV[$((T / 12))]}"
  RUN=${RUNS[$(((T % 12) / 2))]}; S=${SC[$((T % 2))]}
  X=(); SUF=""; [ "${SMOKE:-0}" = 1 ] && { X=(--smoke --nb 3 --boot 20); SUF=_smoke_nb3; }
  [ -s results/r3/8/p2d/cells/${DS}_${VAR}_${RUN}_${S}${SUF}.json ] && { echo done; exit 0; }
  $PY scripts/r3/item8_p2d_score.py --ds $DS --variant $VAR --run $RUN --scorer $S "${X[@]}" ;;
report)
  $PY scripts/r3/item8_p2d_report.py ;;
*) echo "MODE?"; exit 2 ;;
esac
