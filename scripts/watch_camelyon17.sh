#!/usr/bin/env bash
# Watch Camelyon17 download/pilot; wilds x3 then wget -c resume.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
PY="/data2/cmdir/home/toandq/.conda/envs/torch-env/bin/python"
LOG="${ROOT}/logs/camelyon17_watchdog.log"
DL_LOG="${ROOT}/logs/camelyon17_download.log"
WGET_LOG="${ROOT}/logs/camelyon17_wget.log"
PILOT_LOG="${ROOT}/logs/camelyon17_pilot.log"
MARKER="${ROOT}/data/raw/wilds/camelyon17_v1.0/RELEASE_v1.0.txt"
SUMMARY="${ROOT}/outputs/reports/camelyon17/seed42/pilot_summary.csv"
TRAIN_FRAC="${TRAIN_FRAC:-0.05}"
WILDS_MAX_ATTEMPTS=3
WGET_MAX_ATTEMPTS=10

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

submit_wilds_download() {
  local attempt=$1
  log "Submitting WILDS download job (attempt ${attempt}/${WILDS_MAX_ATTEMPTS})..."
  nohup srun --gres=gpu:0 -c 4 --mem=16G -t 06:00:00 \
    --job-name=dst-camelyon-dl \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      ${PY} scripts/camelyon17_pilot.py --stage download --download --batch-size 8 --num-workers 2
      ${PY} scripts/camelyon17_pilot.py --stage probe
    " >> "${DL_LOG}" 2>&1 &
}

submit_wget_download() {
  local attempt=$1
  log "Submitting wget -c download job (attempt ${attempt}/${WGET_MAX_ATTEMPTS}) — keeping partial archive..."
  nohup srun --gres=gpu:0 -c 4 --mem=16G -t 08:00:00 \
    --job-name=dst-camelyon-dl \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      bash scripts/download_camelyon17_wget.sh
      ${PY} scripts/camelyon17_pilot.py --stage probe
    " >> "${WGET_LOG}" 2>&1 &
}

submit_pilot() {
  log "Submitting GPU pilot job..."
  nohup srun --gres=gpu:1 -c 8 --mem=32G -t 24:00:00 \
    --job-name=dst-camelyon-pilot \
    bash -lc "
      set -e
      cd ${ROOT}
      export PYTHONPATH=.
      MARKER=${MARKER}
      while [ ! -f \"\${MARKER}\" ]; do sleep 120; done
      ${PY} scripts/camelyon17_pilot.py --stage all --backbone all --seed 42 \
        --batch-size 64 --num-workers 4 --train-frac ${TRAIN_FRAC}
    " >> "${PILOT_LOG}" 2>&1 &
}

wilds_failures=0
wget_failures=0
use_wget=0
pilot_submitted=0

# Count completed WILDS failures (each leaves one "may be corrupted" block in DL log).
if [ -f "${DL_LOG}" ]; then
  wilds_failures=$(grep -c "may be corrupted" "${DL_LOG}" 2>/dev/null || true)
  wilds_failures=${wilds_failures:-0}
  if [ "${wilds_failures}" -ge "${WILDS_MAX_ATTEMPTS}" ]; then
    use_wget=1
  fi
fi

log "Watchdog started (wilds max ${WILDS_MAX_ATTEMPTS}, then wget -c). prior_wilds_failures=${wilds_failures} use_wget=${use_wget}"

while [ ! -f "${SUMMARY}" ]; do
  if [ -f "${MARKER}" ] && [ "${pilot_submitted}" -eq 0 ]; then
    if ! squeue -u "$(whoami)" -n dst-camelyon-pilot -h 2>/dev/null | grep -q .; then
      submit_pilot
      pilot_submitted=1
    fi
  fi

  if [ ! -f "${MARKER}" ]; then
    if squeue -u "$(whoami)" -n dst-camelyon-dl -h 2>/dev/null | grep -q .; then
      sz=$(stat -c%s "${ROOT}/data/raw/wilds/camelyon17_v1.0/archive.tar.gz" 2>/dev/null || echo 0)
      mode=$([ "${use_wget}" -eq 1 ] && echo wget || echo wilds)
      log "Download running (${mode})... archive ${sz} bytes"
    elif [ "${use_wget}" -eq 1 ]; then
      if [ -f "${WGET_LOG}" ] && tail -3 "${WGET_LOG}" 2>/dev/null | grep -q "Camelyon17 wget download OK"; then
        : # extract done, wait for probe / marker
      elif [ -f "${WGET_LOG}" ] && grep -qE "ERROR:|still incomplete|Exited with exit code 1" "${WGET_LOG}" 2>/dev/null; then
        wget_failures=$((wget_failures + 1))
        if [ "${wget_failures}" -ge "${WGET_MAX_ATTEMPTS}" ]; then
          log "wget download failed ${WGET_MAX_ATTEMPTS} times — giving up."
          exit 1
        fi
        log "wget failed — retry ${wget_failures}/${WGET_MAX_ATTEMPTS} (resume with -c)"
        submit_wget_download "${wget_failures}"
        sleep 30
      else
        log "Switching to wget -c after WILDS failures (keep partial archive)."
        submit_wget_download $((wget_failures + 1))
        sleep 30
      fi
    elif grep -qE "IncompleteRead|may be corrupted|FileNotFoundError|Exited with exit code 1" "${DL_LOG}" 2>/dev/null; then
      wilds_failures=$(grep -c "may be corrupted" "${DL_LOG}" 2>/dev/null || true)
      wilds_failures=${wilds_failures:-0}
      log "WILDS download failed (${wilds_failures}/${WILDS_MAX_ATTEMPTS})"
      if [ "${wilds_failures}" -ge "${WILDS_MAX_ATTEMPTS}" ]; then
        use_wget=1
        log "Switching to wget -c (resume partial archive, no delete)."
        submit_wget_download 1
      else
        rm -f "${ROOT}/data/raw/wilds/camelyon17_v1.0/archive.tar.gz"
        submit_wilds_download "${wilds_failures}"
      fi
      sleep 30
    elif [ ! -f "${ROOT}/data/raw/wilds/camelyon17_v1.0/archive.tar.gz" ]; then
      submit_wilds_download 1
      sleep 30
    fi
  fi

  if squeue -u "$(whoami)" -n dst-camelyon-pilot -h 2>/dev/null | grep -q .; then
    pilot_submitted=1
  elif [ -f "${MARKER}" ] && [ ! -f "${SUMMARY}" ] && [ "${pilot_submitted}" -eq 1 ]; then
    if grep -qE "Traceback|Exited with exit code 1" "${PILOT_LOG}" 2>/dev/null; then
      if ! tail -5 "${PILOT_LOG}" 2>/dev/null | grep -q "pilot done"; then
        log "Pilot failed — resubmitting..."
        submit_pilot
        sleep 30
      fi
    fi
  fi

  sleep 180
done

log "DONE — pilot_summary.csv ready at ${SUMMARY}"
cat "${SUMMARY}" >> "${LOG}"
