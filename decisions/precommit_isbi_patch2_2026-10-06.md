# Precommit: ISBI patch 2 — ResNet50 diagnosis, balanced folds, more seeds, same-protocol H5(a)

Date: 2026-10-06 (written 02:50 +07). Branch `rigor-pack`, after `c0fba53`. Committed before any
patch-2 training, scoring or diagnosis number is computed. Not edited after results; deviations go
to a dated "Deviations" section appended at the end, with the reason.

Unchanged: every bar, seed, permutation count and split of `decision_precommit_rigor_pack.md`,
the numerical-stability addendum (Ledoit-Wolf float64 Mahalanobis), `precommit_sample_size_2026-10-05.md`
and `precommit_isbi_patch_2026-10-05.md` (patch 1; its folds, numbers and bars stay as reported).

Known before writing (already pushed): all patch-1 results (`7b772d3`) and the patch-1 per-fold
table / ResNet50 diagnostic (`c0fba53`: no sign flip found, loss curve monotone, best-checkpoint
selection accuracy 0.995-0.997, ResNet50 predicts tumour on <1% of hospital-2 patches). Known
structure (metadata only): train split = 3 hospitals x 10 slides; per-slide train patch counts
1,430-55,149. Not known: anything trained or scored on the balanced folds, seeds 43/44 of the
retrain, any feature score of a retrained model, fold-0 class balance, last-epoch checkpoints.

## P1. ResNet50 AUROC 0.27 diagnosis (CPU first, no threshold change)

Checks, in this order, on the patch-1 ResNet50 runs (seed 42, folds 0 and 1):
- (a) score sign and ID/OOD label orientation in `logit_retrain_slide_disjoint.py` (MSP/Energy
  higher = ID; `calc_auroc` ID = 1); recompute AUROC from the saved logits independently.
- (b) per-fold accuracy and MSP/Energy AUROC on seen vs unseen id_val slides, plus OOD accuracy and
  predicted-class fractions.
- (c) training collapse: loss curve, constant predictions, tumour fraction of each fold's train
  patches.
- (d) checkpoint selection: best-epoch vs last epoch (patch 1 saved only the best checkpoint;
  best/last selection accuracies from the log; the v2 runs below save both, and the last-epoch
  AUROC is reported as a diagnostic, never as the primary number).
Rule: a technical bug (wrong sign, label/index misalignment, wrong subset, wrong checkpoint loaded)
→ fix it in a separate "fix:" commit, rerun only the affected patch-1 fold(s), report old and fixed
numbers in the README. No bug → keep 0.27, label it "anomalous fold" (classifier collapse on the OOD
hospital) in every table. Hyperparameters never change.
Output: `outputs/reports/rigor_pack/isbi_patch2/resnet50_diagnosis.md`.

## P2. Balanced folds (fixed now)

Within each train hospital (10 slides each), enumerate every split of its slides into 5 + 5 and keep
the splits minimising |train patches fold 0 − train patches fold 1| of that hospital. If several
splits (including mirrored ones) tie, pick one with `numpy.random.default_rng(20261006)`
(`rng.integers(len(ties))`, hospitals in ascending id order, candidates in lexicographic order of
the fold-0 slide tuple). id_val patches follow their slide. The fold sizes are printed and written
to the README; the assignment is deterministic and the same for every arch and seed.

## P3. Grid

Retrain (MSP/Energy slide-disjoint, patch-1 protocol) on the balanced folds: ResNet50, ConvNeXt-T,
DenseNet121 × seeds {42, 43, 44} × folds {0, 1} = 18 trainings. Recipe = each arch's
`camelyon17_pilot.py` recipe used for patch 1 and the seed-42 paper run (`resolve_recipe`, 10
epochs, cosine, batch size, transforms), `set_seed(seed)`; checkpoint selected on accuracy on the
id_val patches of the same fold's slides; the last-epoch checkpoint is also saved.
Budget: smoke = ResNet50 seed 42 fold 0 (part of the grid). After it finishes, projected total =
its wall time (training + extraction) × Σ over the 18 runs of the arch factor (patch-1 per-patch
epoch time of the arch / that of ResNet50) × (run's train patches / smoke train patches). If the
projection > 30 GPU-h, seed 44 is dropped (6 runs), decided by this rule only, stated in the
README. <= 4 GPUs at once; node002 excluded; data staged to node-local /tmp (`stage_wilds.sh`).
A missing-data failure is fixed by paths only.

## P4. Metrics and bar (locked)

- Primary: within-model gap = AUROC(seen-slide id_val vs OOD) − AUROC(unseen-slide id_val vs OOD),
  same retrained model, OOD = all hospital 2, per fold; fold mean; mean ± SD (ddof = 1) over seeds
  of the fold means. For MSP and Energy (T = 1).
- Secondary: published seed-42 AUROC − slide-disjoint retrain AUROC (patch-1 metric), same layout.
- Unseen-slide and seen-slide id_val accuracy and OOD accuracy are reported next to every AUROC.
- Bar (unchanged from patch 1, applied to the primary metric): mean-over-seeds gap > 0.02 for MSP
  or Energy on >= 2/3 archs → "logit scores also benefit from slide sharing"; gap <= 0.02 on all 3
  archs and both scores → "MSP/Energy change by <= 0.02 on unseen slides"; otherwise per-arch
  report, no sentence.
- Confound note required in the README and summary if unseen-slide accuracy < 0.8 in any fold.
- A ResNet50 cell with AUROC < 0.5 is labelled "anomalous" (collapse) if P1 found no bug; it is
  never dropped.

## P5. Feature scores on the same retrained models (same protocol)

For each of the retrained models: extract avgpool features / logits (eval transform) for the
fold's train patches, all id_val and all hospital 2. Fit the published scorer
`OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)`
(Ledoit-Wolf float64 Mahalanobis, cosine kNN, ViM, ReAct p90) on the fold's train patches.
ID = unseen-slide id_val (primary) and seen-slide id_val (descriptive). All 7 scores
(MSP, Energy, ELogitNorm, ViM, ReAct, Mahalanobis, kNN).
Per (arch, seed, fold) cell: winner = argmax AUROC over the 7 on unseen-slide ID. Count cells won
by the feature family: primary F3 = {Mahalanobis, kNN, ViM}, secondary F2 = {Mahalanobis, kNN}.
Descriptive, no pass/fail bar; allowed wording "under a same-protocol slide-disjoint comparison,
feature-space scores win K/N cells (F3)". Also reported per arch (cells won out of 6 or 4).
Output: `isbi_patch2/same_protocol_h5a.csv` (all 7 AUROCs per cell + winner).

## Deliverables

`outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint_v2/` (per-run JSON);
`outputs/reports/rigor_pack/isbi_patch2/`: `resnet50_diagnosis.md`, `phaseB_per_fold.csv/.md`,
`same_protocol_h5a.csv`, `README.md` (pass/fail vs each bar, fold sizes, GPU-h estimate vs actual,
bug fixes if any). Checkpoints, logits and feature arrays are not committed; no home paths.
Estimate before submitting: ~16 GPU-h training (3 seed sets × patch-1 5.2 GPU-h, balanced folds have
the same total patches) + ~1-2 GPU-h extraction ≈ 17-18 GPU-h; scoring on CPU.
