#!/bin/bash
# R3 item 8 / P2-d: wait for the report job, then commit data (pilot, train JSONs, cells), regenerate REPORT.md with that
# commit hash, commit and push. Runs on the login node (git push); exits when done or after 72 h. Usage: p2d_finalize.sh <report_jobid>
set -uo pipefail
cd $HOME/DST-Skin
J=$1
for i in $(seq 1 4320); do
  st=$(sacct -j $J -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL) break ;; esac
  sleep 60
done
echo "report job $J: $st $(date)"
msg() { printf '%s\n' "$1" > /tmp/dst_msg_p2d; }
commit() { C=$(git commit-tree $(git write-tree) -p HEAD -F /tmp/dst_msg_p2d) && git update-ref refs/heads/rigor-pack $C HEAD; }
git add results/r3/8/p2d/pilot_ood.json results/r3/8/p2d/pilot_ood.csv results/r3/8/p2d/train results/r3/8/p2d/cells 2>/dev/null
msg "R3 item 8 / P2-d: pilot, training / probe JSONs and scoring cells"
commit
H=$(git rev-parse --short HEAD)
$HOME/.venvs/crossfit-r3/bin/python scripts/r3/item8_p2d_report.py $H
git add results/r3/8/p2d/REPORT.md
msg "R3 item 8 / P2-d: REPORT (data at $H)"
commit
git push -q origin rigor-pack && echo "pushed $(git rev-parse --short HEAD) $(date)"
