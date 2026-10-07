#!/bin/bash
# Quarantine (do not delete, do not open) all MICCAI-campaign outputs after a REPRO mismatch.
# Only file names are recorded; contents stay unread. Restore: chmod 700 "$Q" after the cause is found.
set -euo pipefail
cd "$(dirname "$0")/../.."

for j in $(squeue -u "$USER" -h -o "%A" -n fm-gate-x,fm-gate-s,fm-gate-d,camp-cpu | sort -u); do
  scancel "$j"
done
sleep 30

Q="outputs/_quarantine_repro_fail/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$Q"
SRC=(
  outputs/rigor_pack/foundation_gate/feats
  outputs/rigor_pack/foundation_gate/load
  outputs/rigor_pack/miccai_campaign/scores
  outputs/reports/rigor_pack/foundation_gate
  outputs/reports/rigor_pack/miccai_campaign/cpu_cells
)
for ds in dermamnist isic2019 breakhis kermany; do
  for f in outputs/rigor_pack/medbench/"$ds"/fm_*; do
    [ -e "$f" ] && SRC+=("$f")
  done
done

: > "$Q/MANIFEST.txt"
for s in "${SRC[@]}"; do
  [ -e "$s" ] || continue
  find "$s" -type f >> "$Q/MANIFEST.txt"
  mkdir -p "$Q/$(dirname "$s")"
  mv "$s" "$Q/$s"
done
n=$(wc -l < "$Q/MANIFEST.txt")
chmod 000 "$Q"
echo "quarantined $n files -> $Q (locked)"
