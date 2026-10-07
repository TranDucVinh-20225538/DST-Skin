#!/bin/bash
# R3 item 4: calibrate -> run -> analyze (one process per CPU, BLAS 1 thread each).
set -euo pipefail
cd "$HOME/DST-Skin"
PY=$HOME/.venvs/crossfit-r3/bin/python
for ph in calibrate run analyze; do
  [ "$ph" = calibrate ] && [ -s "$HOME/r3work/item4/calibration.json" ] && [ "${RECAL:-0}" = 0 ] && continue
  $PY scripts/r3/item4_anisotropic.py $ph --workers "$SLURM_CPUS_PER_TASK"
done
