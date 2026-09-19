#!/usr/bin/env bash
# Resume-capable Camelyon17 download (wget -c) + extract for WILDS layout.
set -euo pipefail

ROOT="/data2/cmdir/home/toandq/DST-Skin"
DATA_DIR="${ROOT}/data/raw/wilds/camelyon17_v1.0"
ARCHIVE="${DATA_DIR}/archive.tar.gz"
URL="https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/"
EXPECTED_SIZE=10658709504
LOG="${ROOT}/logs/camelyon17_wget.log"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "=== wget -c Camelyon17 $(date) ===" | tee -a "${LOG}"
echo "Archive: ${ARCHIVE}" | tee -a "${LOG}"

if [ -f RELEASE_v1.0.txt ] && [ -f metadata.csv ]; then
  echo "Already extracted — skip." | tee -a "${LOG}"
  exit 0
fi

# Resume partial download from failed wilds attempts when possible.
wget -c --timeout=120 --tries=0 --retry-connrefused \
  -O "${ARCHIVE}" "${URL}" 2>&1 | tee -a "${LOG}"

SIZE=$(stat -c%s "${ARCHIVE}")
echo "Downloaded size: ${SIZE} / ${EXPECTED_SIZE}" | tee -a "${LOG}"
if [ "${SIZE}" -lt $((EXPECTED_SIZE - 1048576)) ]; then
  echo "ERROR: archive still incomplete (${SIZE} bytes)" | tee -a "${LOG}"
  exit 1
fi

echo "Extracting..." | tee -a "${LOG}"
tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"
rm -f "${ARCHIVE}"

if [ ! -f RELEASE_v1.0.txt ]; then
  echo "ERROR: RELEASE_v1.0.txt missing after extract" | tee -a "${LOG}"
  exit 1
fi

echo "=== Camelyon17 wget download OK $(date) ===" | tee -a "${LOG}"
ls -la "${DATA_DIR}" | head -20 | tee -a "${LOG}"
