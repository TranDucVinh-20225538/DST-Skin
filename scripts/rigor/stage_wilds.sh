# Sourced (not executed) at the top of a GPU job body when DST_STAGE_DATA=1; the main command is
# then run as `stage_run CMD...`.
# Copies Camelyon17 to node-local storage for THIS job only and exports DST_WILDS_ROOT (read by
# src/datasets/camelyon_ood.py). Data and code are unchanged; only the read location moves off
# the shared filesystem. The copy is removed when the job shell exits, on error, scancel or
# timeout: copy/extract and the main command run in the background under `wait`, so the
# TERM trap fires at once instead of after a child stuck in disk I/O. Leftovers of my own
# SIGKILLed jobs are removed by the next staged job and by `stage_wilds.sh cleanup`.
# Target: node-local ${TMPDIR:-/tmp} if it has 2x the space, else the shared copy as before.
# /dev/shm only with DST_STAGE_SHM=1 (and dataset < 1/4 of free RAM): on node004 systemd-logind
# RemoveIPC wipes all of a user's /dev/shm files when one of their jobs ends.
# Source: the tar shards of scripts/rigor/pack_wilds.sh if complete (large sequential reads),
# else the original small files. The copy is used only if its file list and sizes match the
# ORIGINAL data/raw tree and an md5 sample of DST_STAGE_MD5_N files matches the originals.

_st_src="${DST_WILDS_SRC:-data/raw/wilds}/camelyon17_v1.0"
_st_pack="${DST_WILDS_PACK:-data/cache/camelyon17_v1.0_tar}"
_st_dst=""
_st_bg=""

_st_cleanup() {
  [[ -n "${_st_bg}" ]] && kill "${_st_bg}" 2>/dev/null
  if [[ -n "${_st_dst}" && -d "${_st_dst}" ]]; then rm -rf "${_st_dst}"; echo "stage: removed ${_st_dst}"; fi
  _st_dst=""
}

_st_remove_stale() {
  local live d j
  live=" $(squeue -h -u "$(id -un)" -o '%A' 2>/dev/null | tr '\n' ' ') "
  for d in /dev/shm/dst_wilds_* "${TMPDIR:-/tmp}"/dst_wilds_*; do
    [[ -d "${d}" && -O "${d}" ]] || continue
    j="${d##*_}"
    [[ "${live}" == *" ${j} "* ]] || { rm -rf "${d}"; echo "stage: removed stale ${d}"; }
  done
}

# run in background and wait, so a TERM trap is handled immediately
_st_bgwait() { "$@" & _st_bg=$!; wait "${_st_bg}"; local rc=$?; _st_bg=""; return ${rc}; }
# the training code waits forever for a missing dataset; fail instead if the copy disappears
_st_watch() {
  local pid=$1 f="${DST_WILDS_ROOT}/camelyon17_v1.0/metadata.csv"
  while kill -0 "${pid}" 2>/dev/null; do
    [[ -f "${f}" ]] || { echo "stage: staged copy disappeared (${f}), stopping job"; kill "${pid}"; return; }
    sleep 30
  done
}
stage_run() {
  if [[ -z "${_st_dst}" || -z "${DST_WILDS_ROOT:-}" ]]; then _st_bgwait "$@"; return; fi
  "$@" & _st_bg=$!
  _st_watch "${_st_bg}" & local w=$!
  wait "${_st_bg}"; local rc=$?
  kill "${w}" 2>/dev/null; _st_bg=""
  [[ -f "${DST_WILDS_ROOT}/camelyon17_v1.0/metadata.csv" ]] || rc=1
  return ${rc}
}

if [[ "${1:-}" == "cleanup" ]]; then _st_remove_stale; return 0 2>/dev/null || exit 0; fi

trap '_st_cleanup' EXIT
trap '_st_cleanup; exit 143' TERM INT USR1
_st_remove_stale

_st_copy() {
  set -e
  mkdir -p "${_st_dst}"
  if [[ -f "${_st_pack}/COMPLETE" ]]; then
    echo "stage: from tar shards (${_st_pack})"
    ls "${_st_pack}"/shard_*.tar | xargs -P 8 -I{} tar -C "${_st_dst}" -xf {}
  else
    echo "stage: no complete tar pack, copying small files"
    mkdir -p "${_st_dst}/camelyon17_v1.0/patches"
    cp "${_st_src}/metadata.csv" "${_st_src}/RELEASE_v1.0.txt" "${_st_dst}/camelyon17_v1.0/"
    ls "${_st_src}/patches" | xargs -P 8 -I{} cp -r "${_st_src}/patches/{}" "${_st_dst}/camelyon17_v1.0/patches/"
  fi
}

_st_verify() {
  local a b
  (cd "${_st_src}" && find . -type f -printf '%P %s\n' | LC_ALL=C sort) > "${_st_dst}/src.manifest"
  (cd "${_st_dst}/camelyon17_v1.0" && find . -type f -printf '%P %s\n' | LC_ALL=C sort) > "${_st_dst}/dst.manifest"
  cmp -s "${_st_dst}/src.manifest" "${_st_dst}/dst.manifest" || { echo "stage: file list / sizes differ from data/raw"; return 1; }
  cut -d' ' -f1 "${_st_dst}/src.manifest" | shuf -n "${DST_STAGE_MD5_N:-2000}" --random-source=<(yes) > "${_st_dst}/sample"
  a=$(cd "${_st_src}" && xargs -a "${_st_dst}/sample" -d '\n' md5sum | md5sum | cut -d' ' -f1)
  b=$(cd "${_st_dst}/camelyon17_v1.0" && xargs -a "${_st_dst}/sample" -d '\n' md5sum | md5sum | cut -d' ' -f1)
  [[ "${a}" == "${b}" ]] || { echo "stage: md5 sample differs from data/raw"; return 1; }
}

_st_need=$(du -sm "${_st_src}" | cut -f1)
_st_free=$(free -m | awk '/^Mem:/{print $4}')
_st_base=""
_st_tmp="${TMPDIR:-/tmp}"
_st_avail=$(df -Pm "${_st_tmp}" | awk 'NR==2{print $4}')
if [[ "${DST_STAGE_SHM:-0}" == 1 && -d /dev/shm && -w /dev/shm ]] && (( _st_need * 4 < _st_free )); then
  _st_base=/dev/shm
elif (( _st_need * 2 < _st_avail )); then
  _st_base="${_st_tmp}"
fi
echo "stage: dataset ${_st_need} MB, free RAM ${_st_free} MB -> ${_st_base:-none (shared copy)}"

if [[ -n "${_st_base}" ]]; then
  _st_dst="${_st_base}/dst_wilds_${SLURM_JOB_ID:-$$}"
  _st_t0=$(date +%s)
  if _st_bgwait _st_copy && _st_bgwait _st_verify; then
    export DST_WILDS_ROOT="${_st_dst}"
    echo "stage: OK $(wc -l < "${_st_dst}/src.manifest") files, md5 sample ${DST_STAGE_MD5_N:-2000} match," \
         "$(( $(date +%s) - _st_t0 )) s -> DST_WILDS_ROOT=${DST_WILDS_ROOT}"
  else
    _st_cleanup
    echo "stage: copy or verification failed -> using the shared copy"
  fi
fi
