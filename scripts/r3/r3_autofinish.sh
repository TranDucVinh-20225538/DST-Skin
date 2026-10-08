#!/bin/bash
# R3 login-node watcher: when an item's jobs have left the queue, commit its data, regenerate its REPORT.md with that
# commit hash, commit and push (git steps under flock, shared with other finalizers). P2-d: also finalizes when its
# dependency chain can never run (cancels the rest) or at the precommitted hard stop. Exits when every item is done.
# Usage: r3_autofinish.sh   (job IDs below; state in $ST)
set -uo pipefail
cd "$HOME/DST-Skin"
PY=$HOME/.venvs/crossfit-r3/bin/python
ST=$HOME/r3work/autofinish
LOCK=$HOME/r3work/git.lock
mkdir -p "$ST"
HARD_STOP=$(date -d "2026-10-15 23:59" +%s)
P2D_CHAIN="64821 64823 64824 64825 64826 64827"

gone() {  # all given job IDs (incl. array tasks) have left the queue; false if squeue itself fails
  local q j
  q=$(squeue -h -u "$USER" -o %i 2>/dev/null) || return 1
  for j in "$@"; do grep -qE "^$j(_|$)" <<< "$q" && return 1; done
  return 0
}
commit() {  # message
  printf '%s\n' "$1" > "$ST/msg"
  git diff --cached --quiet && return 0
  C=$(git commit-tree "$(git write-tree)" -p HEAD -F "$ST/msg") && git update-ref refs/heads/rigor-pack "$C" HEAD
}
finalize() {  # name, "data paths", "report cmd (gets hash appended)", "report paths" [, fallback cmd on generator failure]
  local name=$1 data=$2 cmd=$3 rep=$4 fb=${5:-}
  (
    flock 9
    git reset -q
    eval "$cmd pending" > "$ST/$name.gen0.log" 2>&1 || echo "generator (pre) failed: $name" >> "$ST/log"
    git add $data 2>/dev/null
    commit "R3 $name: data"
    H=$(git rev-parse --short HEAD)
    if eval "$cmd $H" > "$ST/$name.gen.log" 2>&1; then
      git add $rep $data 2>/dev/null
      commit "R3 $name: REPORT (data at $H)"
    else
      echo "generator failed: $name (see $ST/$name.gen.log)" >> "$ST/log"
      if [ -n "$fb" ]; then
        eval "$fb $H" && git add $rep && commit "R3 $name: STOP REPORT (data at $H)"
      fi
    fi
    git push -q origin rigor-pack && echo "$(date '+%F %T') $name pushed $(git rev-parse --short HEAD)" >> "$ST/log"
  ) 9>"$LOCK"
}
p2d_stop_report() {  # hash; writes a STOP REPORT when the P2-d generator cannot run
  { echo "# R3 item 8 / P2-d"; echo; echo "Commit: $1"; echo
    echo "Verdict: STOP ($P2D_WHY); the report generator could not run on the available outputs."; echo
    echo "Job states (SLURM): $(sacct -j "$(echo $P2D_CHAIN | tr ' ' ,)" -X -n -o JobName%24,State | tr -s ' ' | paste -sd ';')"
  } > results/r3/8/p2d/REPORT.md
}

p2d_stopped() {  # chain broken: a dependant can never run
  squeue -h -j "$(echo $P2D_CHAIN | tr ' ' ,)" -o "%i %r" 2>/dev/null | grep -q DependencyNeverSatisfied
}

echo "$(date '+%F %T') start" >> "$ST/log"
while :; do
  now=$(date +%s)
  if [ ! -e "$ST/p2b.done" ] && gone 64759 64808 64817; then
    finalize "item 8 / P2-b" "results/r3/8/p2b/cells results/r3/8/p2b/cells_ood_clean" \
      "$PY scripts/r3/item8_p2b_report.py" "results/r3/8/p2b/REPORT.md"
    touch "$ST/p2b.done"
  fi
  if [ ! -e "$ST/item1.done" ] && gone 64813 64814; then
    finalize "item 1" "results/r3/1/match.csv results/r3/1/match_table.md results/r3/1/orphan_seen.csv results/r3/1/paper_ci_v2.csv" \
      "$PY scripts/r3/item1_report.py" "results/r3/1/REPORT.md"
    touch "$ST/item1.done"
  fi
  if [ ! -e "$ST/overlap.done" ] && gone 64822 64813; then
    finalize "item 8 (post hoc) OOD overlap" "results/r3/8/ood_overlap/tracka_kermany_clean_ood.csv" \
      "$PY scripts/r3/item8_ood_overlap_report.py" "results/r3/8/ood_overlap/REPORT.md"
    touch "$ST/overlap.done"
  fi
  if [ ! -e "$ST/item2.done" ] && [ -e "$ST/item1.done" ] && gone 64617; then
    finalize "item 2" "results/r3/2" "$PY scripts/r3/item2_report.py" "results/r3/2/REPORT.md"
    touch "$ST/item2.done"
  fi
  if [ ! -e "$ST/p2d.done" ]; then
    why=""
    gone $P2D_CHAIN && why="chain finished"
    [ -z "$why" ] && p2d_stopped && why="dependency never satisfied"
    [ -z "$why" ] && [ "$now" -ge "$HARD_STOP" ] && why="hard stop 2026-10-15"
    if [ -n "$why" ]; then
      P2D_WHY=$why
      echo "$(date '+%F %T') P2-d: $why; states: $(sacct -j "$(echo $P2D_CHAIN | tr ' ' ,)" -X -n -o JobID,State | tr -s ' ' | tr '\n' ';')" >> "$ST/log"
      [ "$why" != "chain finished" ] && scancel $P2D_CHAIN 2>/dev/null
      finalize "item 8 / P2-d" "results/r3/8/p2d/pilot_ood.json results/r3/8/p2d/pilot_ood.csv results/r3/8/p2d/train results/r3/8/p2d/cells" \
        "$PY scripts/r3/item8_p2d_report.py" "results/r3/8/p2d/REPORT.md" p2d_stop_report
      touch "$ST/p2d.done"
    fi
  fi
  n=$(ls "$ST"/{p2b,item1,overlap,item2,p2d}.done 2>/dev/null | wc -l)
  if [ "$n" -eq 5 ]; then
    echo "$(date '+%F %T') all done; r3 jobs left: $(squeue -u "$USER" -h -o %j | grep -c '^r3' )" >> "$ST/log"
    exit 0
  fi
  [ "$now" -ge $((HARD_STOP + 86400)) ] && { echo "$(date '+%F %T') giving up" >> "$ST/log"; exit 1; }
  sleep 120
done
