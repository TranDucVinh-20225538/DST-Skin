# Precommit: group-leakage check across benchmarks (MIDOG, RxRx1-WILDS, iWildCam-WILDS)

Date: 2026-10-06 (written 03:30 +07). Branch `rigor-pack`, after `39f8e55`. Committed before any
AUROC, accuracy or score of this check is computed. Not edited after results; deviations go to a
dated "Deviations" section appended at the end, with the reason.

Question: does ID evaluation data that shares groups (slide / case / experiment / camera location)
with training inflate post-hoc OOD AUROC and reorder detectors, as found on Camelyon17 (H11c,
ISBI patches 1-2)? Protocol, 7 scores and the 0.02 bar are the Camelyon ones. Unchanged: every
earlier precommit and bar; the Ledoit-Wolf float64 Mahalanobis fix; ISBI patch 2 (running).

Known before writing (inventory only, no score computed): see `outputs/reports/rigor_pack/multibench/inventory.md`
(written from the same facts): MIDOG train / id_val / id_test(1a) cases are pairwise disjoint (40 / 5 / 5
cases) and ID is a single scanner (1a); iWildCam seed-42 models exist for all 8 archs, train = 243
locations, all 146 id_val locations occur in train, the 48 OOD-test locations do not; the existing
iWildCam feature caches were extracted with a shuffled train loader (no per-image index, so no
group ID); RxRx1 is not on the HPC (WILDS v1.0, 7.4 GB, direct download, no licence click) and no
RxRx1 model exists. No number from any of these datasets has been read for this question.

## Datasets and groupings (fixed)

| dataset | group | ID (standard) | OOD | archs (A) | archs (B) |
|---|---|---|---|---|---|
| MIDOG (1a) | case (scanner as 2nd) | test 1a | cs-ID 1b+1c | — | — |
| RxRx1-WILDS | experiment (plates nested) | `id_test` (site 2 of train wells) | `test` experiments | R18, R50, DenseNet121, ConvNeXt-T | same 4 |
| iWildCam-WILDS | camera location | `id_val` | `test` (OOD locations) | all 8 seed-42 models | R18, R50, DenseNet121, ConvNeXt-T |

- MIDOG: the standard ID split is already case-disjoint from train and there is one ID scanner, so no
  seen-group ID set exists under this protocol (scanner-disjoint ID is impossible). (A)/(B) are not
  run; MIDOG is recorded as **"no group sharing by construction"** and counts as leakage *absent*
  in the generalization rule. No substitute grouping or dataset is introduced.
- Folds (A and B, all seeds): train groups shuffled with `numpy.random.default_rng(1042)` within
  strata (RxRx1: cell type; iWildCam: one stratum), then assigned in that order greedily to the fold
  with fewer train images so far (tie → fold 0). ID images follow their group; ID images of groups
  without train images (none expected) are excluded and counted. Fold sizes written to the README.

## (A) Scorer-fit leakage (seed 42)

Exactly the Camelyon H11c / patch-1 A protocol: per fold f, `OODScorer(k_nearest=50, use_react=True,
react_percentile=90.0, use_vim=True, vim_dim=None)` fitted on fold-f train images of the standard
(full-train) seed-42 model, features cast to float64 (Ledoit-Wolf float64 Mahalanobis);
same-group AUROC = ID images of fold-f groups vs all OOD; disjoint AUROC = ID images of fold-(1−f)
groups vs all OOD; average over the 2 folds; Δ = disjoint − same. AUROC via `calc_auroc`.
Indexed features (deterministic loader order, per-image group ID) are re-extracted for every model;
existing checkpoints are reused unchanged.
Bar: for a score in {Mahalanobis, kNN, ViM, ReAct}, loss (−Δ) > 0.02 on >= half of the archs →
"leakage present for that score on that dataset".

## (B) Logit leakage (retrain, seeds 42 and 43)

Retrain on fold-f train groups, f = 0, 1; 4 archs × 2 folds × 2 seeds = 16 runs per dataset.
- iWildCam: `iwildcam_pilot.py` recipe unchanged (`cnn_recipe`, 10 epochs, cosine, ImageNet init),
  `set_seed(seed)`, checkpoint by accuracy on id_val images of the same fold's locations.
- RxRx1 (no repo recipe exists; fixed now): `cnn_recipe(arch)` optimiser / lr / wd / batch size,
  10 epochs, cosine, ImageNet init, 224 px resize, train augmentation = horizontal flip + random
  90° rotation, ImageNet normalisation, last-epoch checkpoint (no ID-test or OOD-val selection).
  The (A) base models for RxRx1 are trained with the same recipe on the full train split, seed 42.
- Metrics: within-model gap = AUROC(seen-group ID vs OOD) − AUROC(unseen-group ID vs OOD) for MSP
  and Energy (T = 1), per fold, fold mean, mean ± SD over seeds; unseen-group and seen-group
  accuracy next to every number; confound flag if unseen-group accuracy < 0.8 (expected on RxRx1,
  1,139 classes; flagged, not a reason to drop).
- Bar: gap > 0.02 for MSP or Energy (mean over seeds) on >= 2/3 of the retrained archs (>= 3 of 4)
  → "logit leakage present on that dataset".
- Also descriptive: the 7 scores on each retrained model, fit = fold train groups, ID = unseen groups
  (same-protocol winner, as patch-2 P5).

## (C) Ranking change (seed 42, (A) archs)

Per arch, 7-score AUROC vector under the standard protocol (published scorer fit on all train, all
ID) vs the group-disjoint protocol (Maha/kNN/ViM/ReAct = (A) disjoint 2-fold AUROC; MSP/Energy/
ELogitNorm of the same standard model on the same fold-(1−f) ID subsets, averaged over folds).
Report winner under each, whether it changes, and Kendall tau-b between the two vectors. Descriptive.

## Dataset verdict and generalization rule (locked)

A dataset has "leakage present" if (A) is present for at least one fit score OR (B) passes.
- Present on >= 2 of the 3 datasets → claim "group-level leakage in WILDS-style ID splits".
- Present on exactly 1 → claim names Camelyon17 and that dataset only; the others are negative
  controls; no general claim.
- Present on none → claim stays Camelyon-specific; the 3 are reported as negative controls.
Since MIDOG counts as absent by construction, the general claim requires both RxRx1 and iWildCam.

## Budget and skips

GPU estimate (fixed now): iWildCam indexed extraction 8 models ~1-2 GPU-h; iWildCam (B) 16 runs ≈
8 full-train equivalents ~8-12 GPU-h; RxRx1 4 base + 16 retrain ≈ 12 full-train equivalents of 40,612
images ~5-8 GPU-h; total ~15-22 GPU-h. Smoke per dataset = ResNet18 seed 42 fold 0. If, after both
smokes, the projected total exceeds 40 GPU-h, RxRx1 seed 43 is dropped (rule fixed now) and stated.
QOS <= 4 GPUs shared with ISBI patch 2 and the user's other project; node002 excluded; data staged to
node-local /tmp. If RxRx1 cannot be downloaded or verified, it is skipped and recorded (no substitute
dataset); a missing-data failure is fixed by paths only; hyperparameters are never changed.

## Deliverables

`outputs/reports/rigor_pack/multibench/`: `inventory.md`, `summary.md` + `summary.csv` (dataset × arch ×
score: Δ, gap, unseen-group accuracy, winner standard vs disjoint, tau), `README.md` (pass/fail per bar
per dataset, generalization verdict, GPU-h estimate vs actual, every skipped item with reason).
No home paths, `.pt`, feature arrays or checkpoints committed.
