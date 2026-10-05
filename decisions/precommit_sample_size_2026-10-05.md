# Precommit: how many seeds / architectures / test patients are needed for a stable detector ranking

Date: 2026-10-05. Branch `rigor-pack`. Written and committed before any number of this analysis was
computed. Not to be edited after results are seen; any later deviation goes into a dated
"Deviations" section appended below, with the reason.

Scope: Camelyon17 (ID = id_val, 26 patients, 33,560 patches; OOD = Hospital-2 test split,
9 patients, 85,054 patches), the official 8-CNN zoo (`common.ARCHS`, EffV2-S = the @224 cell),
seeds 42-46, the 7 precommitted scores (`common.METHODS_ORDER`). CPU only. Does not touch the
running GPU jobs, the manuscript, or `decision_precommit_rigor_pack.md`.

## Scores used

- Primary: Mahalanobis = L2-normalise + sklearn LedoitWolf, float64; ViM = the published code
  path run at 8 BLAS threads (repeat 2 of the stability runs). This is the variant tree
  `outputs/rigor_pack/stability/variants/stable_maha_vim8` ("only Mahalanobis changed").
  The other five scores are the as-run values.
- Appendix only: the `stable` tree (Mahalanobis LW float64 + ViM with dim at 90% explained
  variance). Same analyses, reported in an appendix table, never used for the conclusions.

AUROC of every (arch, seed, score) cell is recomputed from the per-patch score caches of the
tree (ID positive, ties counted 1/2, same convention as `transfer_regret_ci.components`). On the
full data this must equal the tree's score_comparison AUROC to 1e-6; the check is reported, and
if it fails the analysis STOPS and is reported, not patched.

## Ranking and stability measures

- Ranking of a data subset: the 7 scores ordered by mean AUROC over all included
  (arch, seed) cells, computed on the included test patients. Reference ranking: same on the full
  data (8 archs x 5 seeds, 26 ID + 9 OOD patients).
- (a) Kendall tau-b between the subset ranking vector (7 mean AUROCs) and the reference vector.
- (b) P(top-1 match): fraction of draws in which the subset's top score equals the reference top
  score (a tie at the top counts 1/size of the tie group if it contains the reference top).
- (c) Kendall W (`common.kendall_w_batch`, tie-corrected) among the raters of the subset:
  - seed axis: W across the k seeds within each arch, mean over the 8 archs (k >= 2);
  - architecture axis: W across the m archs within each seed, mean over the 5 seeds;
  - patient axes: cross-arch W (8 archs) within each seed, mean over the 5 seeds, on the
    subset AUROCs;
  - grid: W across the m archs within each included seed, mean over the k included seeds.
  W is reported (median, IQR), it has no threshold.
- Threshold "reliable" (fixed now): median tau >= 0.8 AND P(top-1 match) >= 0.8.
- Minimum size of an axis: the smallest size strictly below the full size such that this size
  and every larger size below full meet the threshold. The full size reproduces the reference by
  construction (tau = 1), so it does not count; if no sub-full size qualifies the answer is
  "không đạt trong dữ liệu hiện có".

## Resampling

- 1000 draws per subset size, numpy `default_rng(20261005)`, one generator per axis created in
  the fixed order seed, arch, ID patients, OOD patients, grid (generator i = default_rng
  ([20261005, i])). Subsets are drawn uniformly WITHOUT replacement (subsampling). Where the
  number of distinct subsets is below 1000 the 1000 draws repeat subsets; this is kept as is.
- Known limitation, stated now: subsets overlap with the reference data, so the curves measure
  agreement with the current full data, not with the population; they are optimistic near the
  full size.

## Step 1: axes (the other axes held at full)

1. Seeds: k = 1..5 of the 5 seeds; all 8 archs, all patients.
2. Architectures: m = 2..8 of the 8 archs; all 5 seeds, all patients.
3. Test patients, clustered by patient (never by patch), all 8 archs x 5 seeds:
   - ID: n = 2..26 of the 26 ID patients, OOD full;
   - OOD: n = 2..9 of the 9 Hospital-2 patients, ID full. OOD Hospital-2 has only 9 patients,
     so this axis is limited to n <= 8 below full.
4. Grid: k seeds (1..5) x m archs (2..8), all patients; for each cell 1000 draws of (seed subset,
   arch subset). Heatmaps of median tau and P(top-1 match).

## Step 2: variance components

Data: B = 100 patient-cluster bootstrap replicates (ID 26 and OOD 9 patients resampled with
replacement, same draw applied to every cell; `default_rng([20261005, 10])`), AUROC per
(score, arch, seed, replicate): 7 x 8 x 5 x 100 = 28,000 rows.

- Primary model (as requested): AUROC ~ score + (1|arch) + (1|seed) + (1|replicate).
- Secondary model (the components that can change a ranking are the score x factor
  interactions; seeds are not shared across archs, so seed is nested in arch):
  AUROC ~ score + (1|arch) + (1|arch:seed) + (1|replicate) + (1|score:arch)
  + (1|score:arch:seed) + (1|score:replicate).
- Estimator: statsmodels MixedLM, REML, one group with crossed variance components
  (`vc_formula`). Reported: each variance, its share of the total random variance (sum of the
  components + residual), and the convergence flag. pymer4 is not installed; not installed for
  this.

## Step 3: why the Camelyon features are near-singular (descriptive, no hypothesis)

For each of the 8 archs x 5 seeds: the train features (`train_feats` of the feature cache,
302,436 patches) and, for comparison, the CIFAR-10 ResNet18 train features of
`/data2/hpcshared/ood-numstab` (3 OpenOOD v1.5 checkpoints). Float64 empirical covariance
(centred, MLE), `numpy.linalg.eigvalsh`; before and after L2 normalisation
x / (||x|| + 1e-8) as in the stable Mahalanobis. Reported per cell: feature dim, condition number
lambda_max / lambda_min (inf if lambda_min <= 0, lambda_min also reported), effective rank
exp(-sum p log p) with p = max(lambda, 0) / sum, and the number of eigenvalues
< 1e-6 * lambda_max. Seed 42 shown in the main table, all seeds in the CSV.

## Outputs

Small CSVs and PNGs under `outputs/reports/rigor_pack/sample_size/`; code in
`scripts/rigor/sample_size.py`. One commit per step ("analysis: ...").
