#!/usr/bin/env bash
# =============================================================================
# Rigor pack: one entrypoint. Read docs/RIGOR_PACK.md first.
# Precommit (bars): decisions/decision_precommit_rigor_pack.md  -> commit it BEFORE running.
#
# Default (no flags): CPU pipeline on SLURM
#   inline : env check, input inventory
#   sbatch : csv (W tests, 10k perms) | regret | ties | patients | vitread | reciperead
#            scores (CPU array, 40 cells) -> persample -> splits -> mtest -> tables
# Opt-in GPU stages (each is its own sbatch job, can be submitted alone):
#   --extract           indexed re-extraction, all 40 Camelyon cells (array, ~8-15 GPU-h total)
#   --extract-missing   only cells whose original feature cache is missing
#   --leak              slide-excluded / slide-disjoint kNN+Maha (needs --extract output)
#   --vit-seeds         train+extract+analyze ViT-B/16 seeds 43-46 (array, ~27 GPU-h total)
#   --recipe-seeds      M1/M2 matched-recipe runs on seeds 43-46 (array of 24, ~65-70 GPU-h total)
# Opt-in CPU stages:
#   --skin-midog        per-sample stats (CIs, bootstrap W) for skin + MIDOG too
#   --power             re-run the a priori power simulation (result already in precommit)
#
# Other flags:
#   --only "csv patients"   run only these default stages (names above)
#   --no-default            skip the default CPU pipeline (e.g. only submit --vit-seeds)
#   --local                 run CPU stages inline, no sbatch (interactive node); GPU flags still sbatch
#   --dry-run               print sbatch commands, submit nothing
#
# Env overrides: DST_PY (python), DST_PARTITION (default defq), DST_EXCLUDE (e.g. node002),
#                DST_GPU_GRES (default gpu:1), DST_ACCOUNT, DST_QOS
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DST_PY:-$HOME/.conda/envs/vllm/bin/python}"
PART="${DST_PARTITION:-defq}"
GRES="${DST_GPU_GRES:-gpu:1}"
LOGD="${ROOT}/logs"
mkdir -p "${LOGD}"

DO_DEFAULT=1; ONLY=""; LOCAL=0; DRY=0
DO_EXTRACT=0; EXTRACT_MISSING=0; DO_LEAK=0; DO_VIT=0; DO_RECIPE=0; DO_SKM=0; DO_POWER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extract) DO_EXTRACT=1 ;;
    --extract-missing) DO_EXTRACT=1; EXTRACT_MISSING=1 ;;
    --leak) DO_LEAK=1 ;;
    --vit-seeds) DO_VIT=1 ;;
    --recipe-seeds) DO_RECIPE=1 ;;
    --skin-midog) DO_SKM=1 ;;
    --power) DO_POWER=1 ;;
    --only) ONLY="$2"; shift ;;
    --no-default) DO_DEFAULT=0 ;;
    --local) LOCAL=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

want() { [[ ${DO_DEFAULT} -eq 1 ]] && { [[ -z "${ONLY}" ]] || [[ " ${ONLY} " == *" $1 "* ]]; }; }

cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
R="scripts/rigor"

COMMON_SB=(--partition="${PART}" --chdir="${ROOT}" --export=ALL,PYTHONPATH="${ROOT}",PYTHONUNBUFFERED=1)
[[ -n "${DST_EXCLUDE:-}" ]] && COMMON_SB+=(--exclude="${DST_EXCLUDE}")
[[ -n "${DST_ACCOUNT:-}" ]] && COMMON_SB+=(--account="${DST_ACCOUNT}")
[[ -n "${DST_QOS:-}" ]] && COMMON_SB+=(--qos="${DST_QOS}")

# submit NAME DEPS EXTRA_SBATCH_ARGS... -- COMMAND
#   DEPS: job ids, space-separated ("" = none). Prefix with "any:" for afterany (run even if a
#   dependency failed; used by mtest/tables, which tolerate missing inputs).
#   Writes the job body to logs/rigor_jobs/NAME.sh (bash, inspectable) and sbatches it.
#   Echoes the job id (or "DRY<name>" in dry-run).
submit() {
  local name="$1" deps="$2"; shift 2
  local extra=()
  while [[ "$1" != "--" ]]; do extra+=("$1"); shift; done
  shift
  local kind="afterok"
  if [[ "${deps}" == any:* ]]; then kind="afterany"; deps="${deps#any:}"; fi
  local ids
  ids=$(echo ${deps} | tr ' ' '\n' | grep -v '^DRY' | grep -v '^$' | tr '\n' ':' | sed 's/:$//' || true)
  local dep=()
  [[ -n "${ids}" ]] && dep=(--dependency="${kind}:${ids}")
  mkdir -p "${LOGD}/rigor_jobs"
  local job="${LOGD}/rigor_jobs/${name}.sh"
  cat > "${job}" <<JOB
#!/bin/bash
set -eo pipefail
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
echo "start \$(date) \$(hostname) task=\${SLURM_ARRAY_TASK_ID:-none}"
$*
echo "done \$(date)"
JOB
  local args=("${COMMON_SB[@]}" --job-name="rigor-${name}" --output="${LOGD}/rigor_${name}_%A_%a.out" ${dep[@]+"${dep[@]}"} ${extra[@]+"${extra[@]}"})
  if [[ ${DRY} -eq 1 ]]; then
    echo "[dry-run] sbatch ${args[*]} ${job}" >&2
    echo "DRY${name}"
  else
    sbatch --parsable "${args[@]}" "${job}" | cut -d';' -f1
  fi
}

# run_cpu NAME DEPS SBATCH_RESOURCES... -- COMMAND : inline if --local, else sbatch
run_cpu() {
  local name="$1" deps="$2"; shift 2
  local extra=()
  while [[ "$1" != "--" ]]; do extra+=("$1"); shift; done
  shift
  if [[ ${LOCAL} -eq 1 ]]; then
    echo ">>> [local] ${name}: $*" >&2
    if [[ ${DRY} -eq 0 ]]; then bash -c "set -eo pipefail; $*" 2>&1 | tee "${LOGD}/rigor_${name}_local.out" >&2; fi
    echo "local"
  else
    submit "${name}" "${deps}" ${extra[@]+"${extra[@]}"} -- "$@"
  fi
}

echo "ROOT=${ROOT}"; echo "PY=${PY}"; echo "partition=${PART} exclude=${DST_EXCLUDE:-none}"
[[ -x "${PY}" ]] || { echo "python not found: ${PY} (set DST_PY)"; exit 1; }
if ! git -C "${ROOT}" ls-files --error-unmatch decisions/decision_precommit_rigor_pack.md >/dev/null 2>&1; then
  echo "WARNING: decisions/decision_precommit_rigor_pack.md is not committed. Commit it before reading results."
fi

CPU_S=(-c 4 --mem=16G -t 04:00:00)
CPU_M=(-c 16 --mem=64G -t 08:00:00)
GPU=(--gres="${GRES}" -c 8 --mem=48G)

declare -a ALL_JOBS=()
note() { ALL_JOBS+=("$1=$2"); }

# ---------------- default CPU pipeline ----------------
if [[ ${DO_DEFAULT} -eq 1 ]]; then
  if want env;    then "${PY}" ${R}/check_env.py || true; fi
  if want inputs; then "${PY}" ${R}/check_inputs.py | tee "${LOGD}/rigor_inputs.out"; fi

  J_CSV=""; J_PAT=""; J_SC=""; J_PS=""; J_SP=""; J_VR=""
  if want csv; then
    J_CSV=$(run_cpu csv "" "${CPU_M[@]}" -- "${PY} ${R}/w_from_csv.py --n-perm 10000"); note csv "${J_CSV}"
  fi
  if want regret; then
    J_RG=$(run_cpu regret "" "${CPU_S[@]}" -- "${PY} ${R}/transfer_regret.py"); note regret "${J_RG}"
  fi
  if want ties; then
    J_TI=$(run_cpu ties "" "${CPU_S[@]}" -- "${PY} ${R}/w_ties.py"); note ties "${J_TI}"
  fi
  if want reciperead; then
    J_RR=$(run_cpu reciperead "" "${CPU_S[@]}" -- "${PY} ${R}/recipe_seeds.py"); note reciperead "${J_RR}"
  fi
  if want patients; then
    J_PAT=$(run_cpu patients "" "${CPU_S[@]}" -- "${PY} ${R}/hospital2_patients.py"); note patients "${J_PAT}"
  fi
  if want vitread; then
    J_VR=$(run_cpu vitread "" "${CPU_S[@]}" -- "${PY} ${R}/vit_seeds.py"); note vitread "${J_VR}"
  fi
  if want scores; then
    if [[ ${LOCAL} -eq 1 ]]; then
      J_SC=$(run_cpu scores "" -- "${PY} ${R}/build_score_cache.py")
    else
      J_SC=$(submit scores "" "${CPU_M[@]}" --array=0-39 -- "${PY} ${R}/build_score_cache.py --task-id \${SLURM_ARRAY_TASK_ID}")
    fi
    note scores "${J_SC}"
  fi
  if want persample; then
    J_PS=$(run_cpu persample "${J_SC}" "${CPU_M[@]}" -t 24:00:00 -- "${PY} ${R}/persample.py --domain camelyon17 --B 1000"); note persample "${J_PS}"
  fi
  if want splits; then
    J_SP=$(run_cpu splits "${J_SC}" "${CPU_M[@]}" -t 24:00:00 -- "${PY} ${R}/coverage_splits.py"); note splits "${J_SP}"
  fi
  if want mtest; then
    J_MT=$(run_cpu mtest "any:${J_CSV} ${J_PS} ${J_SP}" "${CPU_S[@]}" -- "${PY} ${R}/multiple_testing.py"); note mtest "${J_MT}"
  fi
  if want tables; then
    J_TB=$(run_cpu tables "any:${J_MT:-} ${J_PAT} ${J_VR} ${J_RG:-} ${J_RR:-} ${J_TI:-}" "${CPU_S[@]}" -- "${PY} ${R}/make_tables.py"); note tables "${J_TB}"
  fi
fi

# ---------------- opt-in CPU ----------------
if [[ ${DO_POWER} -eq 1 ]]; then
  J=$(run_cpu power "" "${CPU_M[@]}" -- "${PY} ${R}/power_w.py --n-sim 400 --n-perm 500"); note power "${J}"
fi
if [[ ${DO_SKM} -eq 1 ]]; then
  J_SKC=$(run_cpu skm_scores "" "${CPU_M[@]}" -t 24:00:00 -- "${PY} ${R}/build_score_cache.py --domains skin_isic_pad midog"); note skm_scores "${J_SKC}"
  J=$(run_cpu skm_persample "${J_SKC}" "${CPU_M[@]}" -t 24:00:00 -- "${PY} ${R}/persample.py --domain skin_isic_pad --B 1000 && ${PY} ${R}/persample.py --domain midog --B 1000"); note skm_persample "${J}"
fi

# ---------------- opt-in GPU ----------------
J_EX=""
if [[ ${DO_EXTRACT} -eq 1 ]]; then
  FLAG=""; [[ ${EXTRACT_MISSING} -eq 1 ]] && FLAG="--only-missing"
  "${PY}" ${R}/extract_features_indexed.py --dry-run ${FLAG} || true
  J_EX=$(submit extract "" "${GPU[@]}" -t 08:00:00 --array=0-39 -- "${PY} ${R}/extract_features_indexed.py --task-id \${SLURM_ARRAY_TASK_ID} --verify ${FLAG}")
  note extract "${J_EX}"
fi
if [[ ${DO_LEAK} -eq 1 ]]; then
  # CPU work on the indexed caches; runs after --extract if both are given.
  J=$(submit leak "${J_EX}" "${CPU_M[@]}" -t 24:00:00 -- "${PY} ${R}/leakfree_knn.py --seeds 42 && ${PY} ${R}/hospital2_patients.py"); note leak "${J}"
fi
if [[ ${DO_VIT} -eq 1 ]]; then
  # same recipe as scripts/submit_camelyon_vit.sh (seed 42); fracX/seedS paths never touch seed 42.
  SEEDS_VIT=(43 44 45 46)
  J_VIT=$(submit vit "" "${GPU[@]}" -t 24:00:00 --array=0-3 -- "SEEDS=(${SEEDS_VIT[*]}); S=\${SEEDS[\${SLURM_ARRAY_TASK_ID}]}; echo ViT seed \${S}; ${PY} scripts/camelyon17_pilot.py --stage all --backbone vit_b_16 --seed \${S} --train-frac 1.0 --num-workers 8 --log-msp-epoch")
  note vit "${J_VIT}"
  J=$(submit vitread_after "${J_VIT}" "${CPU_S[@]}" -- "${PY} ${R}/vit_seeds.py"); note vitread_after "${J}"
fi

if [[ ${DO_RECIPE} -eq 1 ]]; then
  # same commands as scripts/submit_camelyon_matched_recipe.sh (seed 42), seeds 43-46.
  # task t: seed = 43 + t / 6 ; job = t % 6 (4 x M1, 2 x M2). Artifacts go to frac1/matched_adam(w)/seedS/.
  J_RC=$(submit recipe "" "${GPU[@]}" -t 24:00:00 --array=0-23 -- "T=\${SLURM_ARRAY_TASK_ID}; S=\$((43 + T / 6)); J=\$((T % 6))
BB=(convnext_tiny mobilenet_v3_large regnet_y_3_2gf efficientnet_v2_s resnet18 resnet50)
B=\${BB[\$J]}; EXTRA=''; [[ \$B == efficientnet_v2_s ]] && EXTRA='--input-size 224'
if [[ \$J -lt 4 ]]; then RC='--artifact-tag matched_adam --optim adam --lr 1e-4 --wd 1e-4'; else RC='--artifact-tag matched_adamw --optim adamw --lr 1e-4 --wd 0.05'; fi
echo recipe \$B seed \$S \$RC \$EXTRA
${PY} scripts/camelyon17_pilot.py --stage all --backbone \$B --seed \$S --train-frac 1.0 --num-workers 8 --log-msp-epoch \$RC --recipe-batch-size 64 \$EXTRA")
  note recipe "${J_RC}"
  J=$(submit reciperead_after "${J_RC}" "${CPU_S[@]}" -- "${PY} ${R}/recipe_seeds.py"); note reciperead_after "${J}"
fi

echo
echo "Submitted / ran:"
for x in "${ALL_JOBS[@]:-}"; do [[ -n "$x" ]] && echo "  $x"; done
echo "Logs: ${LOGD}/rigor_*.out   Reports: outputs/reports/rigor_pack/"
echo "Check REPRO lines first:  grep -h REPRO ${LOGD}/rigor_*.out"
