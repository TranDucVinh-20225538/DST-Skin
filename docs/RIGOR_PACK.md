# Rigor pack: how to run it

This folder of scripts answers the questions a strict reviewer will ask about the MIDL 2027
paper (`manuscript/midl2027/main.tex`). It does **not** change the paper. It does **not**
change official W (0.694 / 0.791 / 0.857). It never overwrites seed-42 files.

- Bars (what counts as pass/fail): `decisions/decision_precommit_rigor_pack.md` (H1–H17).
- The reviewer's list of weaknesses, and the proposed text edits: `docs/REVIEWER_AUDIT.md`.
- Code: `scripts/rigor/`. One entrypoint: `run_rigor_pack.sh` (repo root).

## 0. Rule number one: commit the precommit first

The precommit is only worth something if git shows it was written before the results.

```bash
git add decisions/decision_precommit_rigor_pack.md scripts/rigor run_rigor_pack.sh docs/RIGOR_PACK.md docs/REVIEWER_AUDIT.md .gitignore
git commit -m "Rigor pack: precommit + scripts (no results)"
git push
```

(On branch `rigor-pack` this is already done. Merge or check out that branch on the HPC.)

## 1. One-time setup on the HPC (bright92)

```bash
cd /path/to/DST-Skin            # your clone; the scripts find ROOT by themselves
git fetch && git checkout rigor-pack
export DST_PY=$HOME/.conda/envs/vllm/bin/python   # the default; set it if your env is elsewhere
PYTHONPATH=. $DST_PY scripts/rigor/check_env.py   # numpy/pandas/scipy/sklearn/torch; wilds only for --extract
PYTHONPATH=. $DST_PY scripts/rigor/check_inputs.py --missing-features
```

`check_inputs.py` lists, for every one of the 40 Camelyon cells (8 archs x seeds 42–46),
whether the score CSV, the feature cache (`outputs/features/...`) and the checkpoint exist.
If any feature cache is missing, use `--extract-missing` (section 3).

Optional, on any machine (no data needed, ~2 min CPU):
`PYTHONPATH=. python scripts/rigor/smoke_test.py` → must end with `SMOKE TEST PASSED`.
(It builds a fake repo in /tmp and runs every stage. "REPRO FAIL" lines inside it are
expected, because the data is fake.)

## 2. The default run (CPU only, SLURM)

```bash
bash run_rigor_pack.sh                 # submits the CPU pipeline with dependencies
bash run_rigor_pack.sh --dry-run       # prints the sbatch commands only
bash run_rigor_pack.sh --local         # no SLURM: runs everything inline (interactive node)
DST_EXCLUDE=node002 bash run_rigor_pack.sh   # keep jobs off a drained/bad node
DST_PARTITION=defq                     # default partition; override if needed
```

What it does, in order (job name `rigor-<stage>`, log `logs/rigor_<stage>_<jobid>_<task>.out`):

| stage | script | needs | time (estimate) | output folder (`outputs/reports/rigor_pack/…`) | precommit |
|---|---|---|---|---|---|
| env, inputs (inline) | `check_env.py`, `check_inputs.py` | – | seconds | log only | – |
| csv | `w_from_csv.py --n-perm 10000` | tracked CSVs only | 0.5–2 h, 16 CPU | `w_csv/` | H1 H2 H4 H5a H7(FPR95) H12 + jump tests |
| regret | `transfer_regret.py` | tracked CSVs only | seconds | `transfer_regret/` | H16 |
| ties | `w_ties.py` | tracked CSVs only | seconds | `w_ties/` | H17 |
| reciperead | `recipe_seeds.py` | matched-recipe CSVs | seconds | `recipe_seeds/` | H15 (INCOMPLETE until `--recipe-seeds` ran) |
| patients | `hospital2_patients.py` | WILDS `metadata.csv` (or the tracked frozen split) | minutes | `hospital2/` | H11a,b + split facts |
| vitread | `vit_seeds.py` | ViT score CSVs | seconds | `vit_seeds/` | H13 (says INCOMPLETE until `--vit-seeds` ran) |
| scores | `build_score_cache.py` (array 0–39) | feature caches | 10–40 min per cell, CPU | caches in `outputs/rigor_pack/` (not tracked) | – |
| persample | `persample.py --domain camelyon17` | score caches | 2–8 h | `persample_camelyon17/` | H3 H5b H7(AUPR) H8 H9a H10 |
| splits | `coverage_splits.py` | score caches + metadata | 2–8 h | `coverage_splits/` | H6 H9b + Kendall tau |
| mtest | `multiple_testing.py` | all `pvalues_family.csv` | seconds | `multiple_testing/` | H14 |
| tables | `make_tables.py` | whatever exists | seconds | `tables/`, `fig/fig1b_valsplit.{pdf,png}` | – |

Each step checks that it reproduces numbers already in the paper (lines starting with
`REPRO`). **Read `REPRO` first. If any line says FAIL, stop and read nothing else.**

```bash
grep -h REPRO logs/rigor_csv_*.out logs/rigor_persample_*.out logs/rigor_splits_*.out
```

Run only some stages: `bash run_rigor_pack.sh --only "csv patients"`.
`mtest` and `tables` use `afterany`, so they still run if one earlier job failed (they skip
missing inputs). The others use `afterok`.

Why 10,000 permutations: the Holm correction over ~16 tests needs p-values far below
0.05/16; with 200 permutations the smallest possible p is 1/201 ≈ 0.005, which can never pass.

## 3. Opt-in GPU stages (each one is a separate sbatch job; none runs by default)

| flag | what | GPU cost (A100, from old logs) | output |
|---|---|---|---|
| `--extract` | re-extract features for all 40 Camelyon cells **with patch indices and slide IDs** (the old caches were saved shuffled, without indices, under train augmentation). Array 0–39, `--verify` compares with the old cache. | ≈13 min per cell → 8–15 GPU-h total | `outputs/features/camelyon17/frac1/seed{S}/rigor_indexed/` |
| `--extract-missing` | only cells with no old cache | per missing cell | same |
| `--leak` | slide-excluded kNN and slide-disjoint 2-fold Maha/kNN (CPU job, waits for `--extract`) | CPU, 1–4 h | `leakage/` (H11c) |
| `--recipe-seeds` | M1/M2 matched-recipe runs (ConvNeXt, MobileNetV3, RegNetY, EffV2-S@224 with ResNet-Adam; ResNet-18/50 with ConvNeXt-AdamW) on seeds 43–46, same commands as `scripts/submit_camelyon_matched_recipe.sh`; then `recipe_seeds.py`. Array 0–23. | 2.5–4.3 h per run (old logs) → ≈65–70 GPU-h | `frac1/matched_adam(w)/seed{43..46}/`, then `recipe_seeds/` (H15) |
| `--vit-seeds` | train + extract + analyze ViT-B/16 on seeds 43–46 (same command as the seed-42 run, `--train-frac 1.0 --log-msp-epoch`); then `vit_seeds.py` | ≈6.8 h per seed → ≈27 GPU-h | standard ViT folders `frac1/seed{43..46}/`, then `vit_seeds/` (H13) |

Examples:

```bash
bash run_rigor_pack.sh --no-default --extract --leak     # only the leakage chain
bash run_rigor_pack.sh --no-default --vit-seeds          # only ViT seeds
bash run_rigor_pack.sh --no-default --recipe-seeds       # only M1/M2 seeds
bash run_rigor_pack.sh --extract --leak --vit-seeds --recipe-seeds --skin-midog   # everything
```

Opt-in CPU: `--skin-midog` (score caches + per-sample CIs/bootstrap W for skin and MIDOG),
`--power` (re-run the a priori power simulation; its result is already in the precommit).

GPU settings: `DST_GPU_GRES` (default `gpu:1`), 8 CPU, 48 GB, like the old `srun` scripts.

## 4. After the run

1. Check `REPRO` lines (above).
2. Open `outputs/reports/rigor_pack/tables/` (CSV + LaTeX) and `multiple_testing/pvalues_summary.txt`.
3. Write `outputs/reports/rigor_pack/VERDICTS.md`: one line per H1–H17: verdict word from the
   precommit + the number. Do not change a bar. If a bar was ill-posed, add a dated note to
   the precommit and give both readings.
4. Only then edit the paper, using the proposed edits in `docs/REVIEWER_AUDIT.md`
   (pick the version that matches the verdict). Separate commit.
5. Commit the small report tables (`outputs/reports/rigor_pack/`). Caches
   (`outputs/rigor_pack/`) and `logs/rigor_*` are gitignored.

## 5. Notes and known caveats

- Python 3.8+; numpy, pandas, scipy, scikit-learn, matplotlib; torch only for score caches
  and extraction; `wilds` only for extraction (`check_env.py` tells you).
- Seed-42 AUROC comes from the frozen `architecture_invariance_ranks.csv` (same as the
  existing cross-seed script); seeds 43–46 from the score CSVs.
- The score caches recompute the 7 scores from the feature files with the repo's
  `OODScorer`; `build_score_cache.py` checks they match the CSV AUROCs to 2e-3 (REPRO).
- Old scripts (`scripts/submit_*.sh`) hardcode an absolute `ROOT` with a username. The rigor
  pack does not: it finds ROOT from its own location and uses `DST_PY`.
- Nothing pins a node. node002 was drained in the logs; use `DST_EXCLUDE` if needed.
