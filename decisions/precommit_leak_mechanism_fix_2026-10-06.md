# Precommit: mechanism, predictor and fix for group leakage

Date: 2026-10-06 (written 03:45 +07). Branch `rigor-pack`, after `909f710`. Committed before any number
of this analysis is computed, and before any ISBI-patch-2 (`precommit_isbi_patch2_2026-10-06.md`) or
multibench (`precommit_group_leakage_multibench_2026-10-06.md`) result has been read. Not edited
after results; deviations go to a dated "Deviations" section appended at the end, with the reason.

Unchanged: all earlier precommits and bars; Ledoit-Wolf float64 Mahalanobis; published scorer config
`OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)`;
`calc_auroc` convention. No retraining beyond the models of ISBI patch 2 and multibench.

## Cells, groups, notation

- Datasets: Camelyon17 (required; group = slide; standard ID = id_val; OOD = hospital 2). iWildCam
  (location) and RxRx1 (experiment) only if multibench produced group-indexed features; MIDOG is
  not used (standard ID split is case-disjoint by construction, multibench precommit).
- Models: published seed-42 models with indexed features (Camelyon: all 8 archs of `rigor_indexed/`;
  iWildCam: the 8 multibench re-extractions; RxRx1: the 4 multibench base models). Retrained models
  (Camelyon: ISBI-patch-2 v2, seeds 42-44; others: multibench (B), seeds 42-43) only where stated.
- Folds: Camelyon = H11c folds (`leakfree_fit_scores.folds`, seed 42); others = multibench folds.
- Δ(score) for a fit score s ∈ {Mahalanobis, kNN, ViM, ReAct} = AUROC_same − AUROC_disjoint of the
  H11c / multibench-(A) 2-fold protocol (a positive Δ = inflation). Within-model gap (retrained) =
  AUROC(seen-group ID) − AUROC(unseen-group ID) for MSP / Energy.
- "Reduction" of an effect under a variant = 1 − effect_variant / effect_original, per cell.

## M. Mechanism

M1 Stain / colour ablation (same models, no retraining; re-extraction on GPU).
- Variants: (a) histology (Camelyon): Macenko stain normalisation, own numpy implementation of
  Macenko et al. 2009 with Io = 240, alpha = 1, beta = 0.15 and the standard reference
  HERef = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], maxCRef = [1.9705, 1.0308]; tissue pixel =
  OD > beta in all three channels; patches with < 10% tissue pixels or a failed eigen-decomposition
  are left unchanged and counted. Other datasets: per-image colour standardisation (per channel
  z-score of the image, then mapped to ImageNet mean / std) before the usual transform.
  (b) grayscale: ITU-R 601 luma replicated to 3 channels, before the usual normalisation.
- Archs: ResNet18, ResNet50, DenseNet121, ConvNeXt-T (published seed 42) for Δ (train + ID + OOD
  re-extracted); within-model gap on the retrained seed-42 models of the same dataset (ID + OOD only).
- Effects: Δ for the 4 fit scores (per arch) and within-model gap for MSP / Energy (per retrained
  model). Reduction per cell; summary = median reduction over all cells of a dataset, per variant.
- Bar: median reduction >= 50% under Macenko/colour OR grayscale → "group appearance is the main
  driver"; < 20% under both → "not explained by colour / stain"; otherwise "partial". ID accuracy
  under each variant is reported next to it (a variant that destroys accuracy weakens the reading).

M2 Group decodability. Logistic regression (sklearn `LogisticRegression(C=1.0, max_iter=1000)`, lbfgs,
features standardised on the training split of each CV fold) predicting group ID from frozen
avgpool features of train + standard ID images. Stratified random subsample of 20,000 images
(`default_rng(20261006)`), groups with < 5 images in the subsample excluded and counted;
stratified 5-fold CV by image. Report balanced accuracy and chance (1 / number of groups). All archs.

M3 Nearest-neighbour attribution. For every standard-ID image: fraction of its k = 50 nearest
training images (cosine on L2-normalised features, full train set = published fit set) from the same
group; mean over ID images. Also the chance fraction (group's share of train) and excess
(observed − chance). All archs. Compared with Δ per arch (Spearman, descriptive).

M4 Instance vs group (Camelyon; other datasets where the same artefacts exist).
- Fit scores (published models, 4 fit scores): within fold f, split fold-f train images 50/50 by image
  (`default_rng(20261006)`); fit the scorer on half A. AUROC vs OOD of: seen instance = half A,
  unseen instance of seen group = id_val of fold-f groups, unseen group = id_val of fold-(1−f) groups.
- Logit scores (retrained seed-42 models): seen instance = fold-f train images (trained on), unseen
  instance of seen group = fold-f id_val, unseen group = fold-(1−f) id_val; MSP / Energy.
- R = (AUROC_unseen-inst-seen-group − AUROC_unseen-group) / (AUROC_seen-inst − AUROC_unseen-group),
  per cell, median per dataset. R >= 0.5 → "group-level effect"; R < 0.5 → "instance memorisation";
  cells with |denominator| < 0.01 are excluded and counted. (On Camelyon the standard id_val patches
  are never trained on, so the measured Δ is by construction not exact-instance memorisation; M4
  asks how much of the seen-instance advantage already holds for new patches of a seen slide.)

Main-driver sentence = M1 verdict, qualified by M4 (group vs instance).

## P. Predictor

Candidates (locked): M2 balanced accuracy; M3 same-group neighbour fraction (raw); centroid distance =
mean over groups of the cosine distance between the group's train-feature centroid and its ID-feature
centroid (L2-normalised features). Unit = (dataset, arch, fit score) cell, target = Δ (seed 42).
Spearman ρ of each candidate with Δ over all cells; leave-one-dataset-out: linear fit Δ ~ candidate on
the other datasets, MAE on the held-out one, averaged over held-out datasets.
Bar: |ρ| >= 0.6 AND LODO MAE < 0.03 → "Δ is predictable from <candidate>"; otherwise "not predictive".
With fewer than 2 datasets LODO is impossible and the verdict is "not testable across datasets"
(ρ reported within Camelyon only).

## F. Fixes (all evaluated, none chosen after results)

Reference F1 (protocol fix) = group-disjoint ID: fit scores = (A) disjoint 2-fold AUROC; MSP / Energy /
ELogitNorm = same model on the same fold-(1−f) ID subsets, fold-averaged. 7-score vector per (dataset, arch).
- F2 cross-fitted scorer, full ID set kept: each ID image is scored by the scorer fitted on the fold
  that does not contain its group; OOD images by the mean of the two fold scorers; one pooled AUROC
  over all ID. (Expected close to F1 by construction; it is the variant that keeps the whole ID set.)
- F3 group-centred features (fit scores recomputed on centred features, standard protocol: full-train
  fit, all ID): (a) per-group centring, each image minus the mean feature of its own group within its
  split (train / ID / OOD groups); (b) per-batch centring, random batches of 256 within each split
  (`default_rng(20261006)`), each image minus its batch mean; (c) M1 Macenko / colour-standardised
  features (4 M1 archs only).
- F4 kNN with same-group exclusion: ID images ignore training neighbours of their own group (OOD
  unchanged), standard protocol otherwise; affects kNN only.
For every fix the other scores keep their standard-protocol values.
Metrics per (dataset, arch): |AUROC_fix − AUROC_F1| over the cells the fix changes; Kendall tau-b
between the fix's 7-score vector and F1's. Per dataset: median |diff| over cells, median tau over
archs; also the fix's change in OOD AUROC relative to the standard protocol (the benchmark's OOD split).
Bar: a fix "works" if median |diff| <= 0.02 AND median tau >= 0.8 on >= 2/3 of the available datasets
(with one dataset: "works on Camelyon only").

## Checklist (deliverable)

5 lines: group-disjoint ID split; Ledoit-Wolf float64; fixed BLAS threads; >= N seeds × M archs from
`sample_size_disjoint/`; report the group-leakage statistic from P (or state that none is predictive).

## Budget, order, outputs

Order: M1-M4 on Camelyon (smoke ResNet18), all archs, then other datasets once multibench features
exist; then P; then F. GPU only for M1 re-extraction: Camelyon ~3 GPU-h (4 models × 2 variants × 420k
images + 6 retrained × 2 × 118k), other datasets ~2-3 GPU-h; total ~5-6 GPU-h. QOS <= 4 GPUs shared with
patch 2 / multibench / the user's other project; node002 excluded; data staged to node-local /tmp.
Outputs: `outputs/reports/rigor_pack/mechanism_fix/` — mechanism.md/.csv, predictor.md/.csv,
fixes.md/.csv, checklist.md, README.md (pass/fail per bar, GPU-h estimate vs actual, skipped items).
No home paths, `.pt`, feature arrays committed.

## Execution order

Set by the user on 2026-10-06 03:08: ISBI patch 2 first, then multibench, then this analysis.
