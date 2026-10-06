#!/bin/bash
# One-time: pack data/raw/wilds/camelyon17_v1.0 into N tar shards under data/cache/ (gitignored),
# so GPU jobs with DST_STAGE_DATA=1 can copy a few large files instead of ~456k small ones
# (the shared filesystem reads small files at < 1 MB/s). Shards are written to *.tmp and renamed;
# the COMPLETE marker is written last, so stage_wilds.sh never uses a partial pack.
# Usage: bash scripts/rigor/pack_wilds.sh [N_SHARDS=8] [DATASET_DIR=camelyon17_v1.0]
set -eo pipefail
cd "$(dirname "$0")/../.."
N=${1:-8}
DS=${2:-camelyon17_v1.0}
SRC=data/raw/wilds
OUT=data/cache/${DS}_tar
mkdir -p "${OUT}"
rm -f "${OUT}"/COMPLETE "${OUT}"/*.tmp
t0=$(date +%s)
if [[ ${DS} == camelyon17_v1.0 ]]; then
  (cd "${SRC}" && ls camelyon17_v1.0/patches) | awk -v n="${N}" '{print > ("'"${OUT}"'/list_" (NR-1)%n)}'
  for i in $(seq 0 $((N - 1))); do
    sed -i 's|^|camelyon17_v1.0/patches/|' "${OUT}/list_${i}"
  done
  echo camelyon17_v1.0/metadata.csv >> "${OUT}/list_0"
  echo camelyon17_v1.0/RELEASE_v1.0.txt >> "${OUT}/list_0"
else
  (cd "${SRC}" && find "${DS}" -type f ! -name archive.tar.gz | LC_ALL=C sort) | awk -v n="${N}" '{print > ("'"${OUT}"'/list_" (NR-1)%n)}'
fi
for i in $(seq 0 $((N - 1))); do
  tar -C "${SRC}" -cf "${OUT}/shard_${i}.tar.tmp" -T "${OUT}/list_${i}" &
done
wait
for i in $(seq 0 $((N - 1))); do mv "${OUT}/shard_${i}.tar.tmp" "${OUT}/shard_${i}.tar"; done
(cd "${SRC}/${DS}" && find . -type f -printf '%P %s\n' | LC_ALL=C sort) > "${OUT}/manifest.txt"
n_tar=$(for i in $(seq 0 $((N - 1))); do tar -tf "${OUT}/shard_${i}.tar"; done | grep -vc '/$')
n_src=$(wc -l < "${OUT}/manifest.txt")
[[ ${n_tar} -eq ${n_src} ]] || { echo "pack: tar has ${n_tar} files, source ${n_src}"; exit 1; }
echo "${N} shards, ${n_src} files, $(( $(date +%s) - t0 )) s, $(date)" > "${OUT}/COMPLETE"
cat "${OUT}/COMPLETE"
