#!/bin/bash
# R3 item 2. MODE=main: pilot -> calibrate -> (submit realdim) -> truth -> cover at the v3 default d.
#            MODE=realdim: Camelyon design, Mahalanobis + kNN, d = $NF, ICC levels from the main calibration.
set -euo pipefail
cd "$HOME/DST-Skin"
PY=$HOME/.venvs/crossfit-r3/bin/python
S=scripts/r3/item2_coverage_real.py
N=$SLURM_CPUS_PER_TASK
B=$HOME/r3work/item2
if [ "${MODE:-main}" = main ]; then
  O=$B/main; mkdir -p $O
  [ -s $O/calibration.json ] || { $PY $S pilot --out $O --workers $N; $PY $S calibrate --out $O --workers $N; }
  for nf in 768 2560; do
    R=$B/realdim_$nf; mkdir -p $R
    $PY -c "import json; c=json.load(open('$O/calibration.json')); json.dump({'camelyon': {k: v for k, v in c['camelyon'].items() if k in ('mahalanobis', 'knn')}}, open('$R/calibration.json', 'w'), indent=1)"
    MODE=realdim NF=$nf sbatch --export=ALL -J r3-2-realdim-$nf -c 24 --mem=96G -t 24:00:00 --nice=8000 --exclude=node002 \
      -o $HOME/DST-Skin/logs/r3_2_realdim_${nf}_%j.out scripts/r3/run_item2.sh
  done
  $PY $S truth --out $O --workers $N
  $PY $S cover --out $O --workers $N
else
  R=$B/realdim_$NF
  # SCORERS restricted through the calibration file (cells are built from it)
  $PY $S truth --out $R --workers $N --designs camelyon --n-features $NF --scorers mahalanobis,knn
  $PY $S cover --out $R --workers $N --designs camelyon --n-features $NF --scorers mahalanobis,knn
fi
