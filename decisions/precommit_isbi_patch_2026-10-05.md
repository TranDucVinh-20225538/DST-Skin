# Precommit: ISBI patch — leakage of fit-based / logit scores, leak-free H5(a), disjoint sample size

Date: 2026-10-05 (written 21:50 +07). Branch `rigor-pack`, after `d503cb6`. Committed before any
Phase A or Phase B number is computed or read. Not edited after results; deviations go to a dated
"Deviations" section appended at the end, with the reason.

Unchanged: every H1–H17 bar, seed, permutation count and split of
`decision_precommit_rigor_pack.md`; the numerical-stability addendum (Ledoit-Wolf float64
Mahalanobis stays the numerical fix; this file does not reopen H1 float32); the frozen manuscript
numbers; `precommit_sample_size_2026-10-05.md`. No new domain, no ViT, no new seeds.

Known before writing (already pushed): H11(c) Maha/kNN slide-disjoint loss > 0.02 on 8/8 archs;
H5(a) re-check with leak-free Maha/kNN = 4/8 (seed 42); H15 per-recipe seed counts; the
overlapping-subset sample-size curves. Not known: any ViM/ReAct slide-disjoint number, any
slide-disjoint retrain number, any disjoint-partition stability number.

## A. ViM / ReAct slide-disjoint 2-fold (extends H11c)

- Folds: exactly the H11c folds of `scripts/rigor/leakfree_knn.py` (train slides grouped by
  hospital, shuffled with `numpy.random.default_rng(1000 + seed)`, alternating fold 0/1).
  REPRO check: fold train sizes must be 97,099 and 205,337 patches as in the H11c log; otherwise
  STOP and report.
- Cells: Camelyon17, all 8 archs, seed 42, indexed caches `rigor_indexed/`.
- Scorer: `OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True,
  vim_dim=None)`, the published configuration of `camelyon17_pilot.py analyze`, fitted on the
  fold's train patches (features, logits, fc). 8 BLAS threads (the "published ViM at 8 threads"
  reading of the stability addendum).
- Per fold f: slide-disjoint AUROC = ID id_val patches whose slide is in fold 1−f vs all
  hospital-2 patches; same-slides AUROC = id_val patches whose slide is in fold f vs all
  hospital 2 (identical to the `*_same_slides` columns of H11c). Average over the 2 folds.
  Delta = slide-disjoint − same-slides.
- Bar (identical to H11c): any arch losing > 0.02 AUROC for ViM or ReAct → the appendix reports
  the slide-disjoint AUROCs of that score and H5(a) uses them; every |delta| <= 0.02 → one
  sentence "ID-side slide sharing changes <score> by <= 0.02".
- MSP / Energy / ELogitNorm have no scorer fitted on train features. Phase A makes no scorer-fit
  leakage claim for them; Phase B covers training-slide memorisation.
- Output: `outputs/reports/rigor_pack/leakage/leakfree_vim_react.csv/.txt` (new file;
  `leakfree_knn.csv` is not rewritten). Smoke: ResNet18 first, then all 8.

## B. Pure-logit memorisation sensitivity (Phase B, opt-in GPU)

- Grid (fixed now, no arch dropped after seeing a delta): seed 42; ResNet50, ConvNeXt-T,
  DenseNet121; folds 0 and 1 of section A (fold sizes are unequal: ~1/3 and ~2/3 of the train
  patches; kept as is so that A and B share folds). 3 archs x 2 folds = 6 GPU runs, <= 4 GPUs at
  once.
- Training: `camelyon17_pilot.py` seed-42 recipe for that arch unchanged (`cnn_recipe`, epochs,
  cosine schedule, batch size, transforms, `set_seed(42)`), train patches restricted to the
  fold's slides. Checkpoint selection by accuracy on the id_val patches of the SAME fold's slides
  (never the evaluated slides).
- Evaluation: MSP and Energy (T = 1) AUROC, ID = id_val patches of the other fold's slides,
  OOD = all hospital 2; average over the 2 folds.
- Primary bar: delta = AUROC_published(seed 42, full train, all id_val) − AUROC_slide_disjoint_
  retrain. |delta| > 0.02 for MSP or Energy on >= 2/3 archs → the paper must not present logit vs
  feature as a fair same-protocol comparison without noting training-slide overlap; |delta| <= 0.02
  on all 3 archs and both scores → one sentence "MSP/Energy change by <= 0.02 under slide-disjoint
  retraining on this grid"; otherwise (1/3 archs) the deltas are reported per arch, no sentence.
- Secondary (descriptive, no bar): within the same retrained model, same-slide id_val (fold f) vs
  other-fold id_val AUROC; this separates slide memorisation from the halved training set.
- Budget: estimate before submitting all 6 = 9-12 GPU-h (from the seed-43..46 recipe runs:
  ResNet50 ~3 h, ConvNeXt ~3.6 h per full training; two folds = one full training). Smoke =
  ResNet50 fold 0; after its first epoch the total is re-estimated from its epoch time; if the
  estimate exceeds 12 GPU-h, Phase B stops, a limitation is written, the grid is not expanded.
  Estimate vs actual is reported in the README.
- Data staged to node-local /tmp with the existing `stage_wilds.sh` watchdog pattern. A failure on
  missing data is fixed by paths only; hyperparameters are never changed.
- Output: `outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint/` (JSON/CSV);
  checkpoints under `data/models/camelyon17/frac1/slide_disjoint/seed42/` (not committed).

## C. H5(a) re-rank (seed 42, 8 archs)

For each arch: argmax AUROC over the scores with a valid number; count cells where the winner is
in the feature family.
- (i) original as-run scores (published cube).
- (ii) Maha/kNN replaced by their H11c slide-disjoint 2-fold AUROCs (= 4/8, already known).
- (iii) Maha/kNN/ViM/ReAct replaced by their slide-disjoint 2-fold AUROCs; MSP/Energy/ELogitNorm
  original.
- (iv) only if Phase B ran: as (iii) and, for the 3 Phase B archs, MSP/Energy replaced by their
  slide-disjoint retrain AUROCs (reported on those 3 cells only).
- Feature family: primary = {Mahalanobis, kNN, ViM} (ViM counted as feature-space when its
  leak-free number exists, i.e. (iii)/(iv); in (i) reported both with and without ViM);
  secondary = {Mahalanobis, kNN} (the original H5 definition). Both always reported.
- Claim language locked: never "feature always best". Allowed: "feature-space wins on K/8
  backbones under slide-disjoint scoring" with K from (iii), primary family.

## D. Disjoint-partition sample size (CPU, no training)

Same data, scores and AUROC conventions as `precommit_sample_size_2026-10-05.md` (primary tree
`stable_maha_vim8`, appendix `stable`; patient-pair U matrices already computed). The subset
ranking is compared with the ranking of the DISJOINT complement instead of the full data:
- seeds: every subset A of k = 1..4 seeds vs the other 5−k seeds (all 8 archs, all patients);
- archs: every subset of m = 2..6 archs vs the other 8−m archs (all seeds, all patients);
- grid: every (k seeds x m archs) block vs the block of the other seeds x other archs,
  k = 1..4, m = 2..6;
- ID patients: n = 2..24 of 26 vs the other 26−n (OOD all 9), 1000 random draws,
  `default_rng([20261005, 20])`;
- OOD patients: every subset of n = 2..7 of the 9 vs the other 9−n (ID all 26).
Seeds/archs/OOD subsets are enumerated exhaustively (deterministic); ID patients are sampled.
Measures and bar unchanged: Kendall tau-b, P(top-1 match), Kendall W within the subset;
"reliable" = median tau >= 0.8 AND P(top-1 match) >= 0.8; minimum size = smallest size in range such
that it and every larger size in range are reliable, else "không đạt trong dữ liệu hiện có".
Rows are labelled `disjoint_partition`; the old rows `overlapping_subset`.
Rule: where the disjoint analysis fails the bar at a size where the overlapping one passed, the
manuscript must prefer the disjoint numbers and call the overlapping curves optimistic.
Output: `outputs/reports/rigor_pack/sample_size_disjoint/`.

## E. H15 checklist

Robust recipe = >= 4/5 seeds with MSP jump >= 0.15 (existing H15 rule, read from
`recipe_seeds/recipe_seeds_summary.txt`). `outputs/reports/rigor_pack/isbi_patch/checklist_h15.md`
(+ .csv) lists only the robust recipes for any deployment-facing sentence; the others are listed
with their count as "do not cite as robust".

## F. Scope table

One table for Limitations: Camelyon17 seed-42 leak-free deltas (Maha, kNN, ViM, ReAct; MSP/Energy
if Phase B ran) and the MIDOG H12 one-liner already computed. No new MIDOG job, no claim beyond
these setups.

## Deliverables

`outputs/reports/rigor_pack/isbi_patch/README.md` (pass/fail of A, B, C, D, E vs the bars; GPU-h
estimate vs actual), optional `claim_edits.md` (suggestions only; manuscript not edited).
No home paths, no `.pt`, no `splits_long.csv.gz` committed.
