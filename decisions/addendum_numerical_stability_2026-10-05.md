# Addendum: numerical stability of Mahalanobis / ViM — written 2026-10-04, after the precommit

**Status: post-precommit, supplementary analysis.** `decisions/decision_precommit_rigor_pack.md`
is unchanged; its hypotheses, bars, permutation counts, splits and seeds are not touched. This
file adds a sensitivity analysis that the precommit did not foresee. Every H1–H17 verdict is
reported under the precommitted pipeline; this addendum only adds a second reading and a rule
for marking verdicts that do not survive it.

## Why

The rigor-pack stage `scores` (`build_score_cache.py`, 2026-10-03) re-fitted the published
scorers on the existing feature caches and compared AUROC with the tracked CSVs. MSP, Energy,
ELogitNorm, ReAct and kNN reproduced to 4 decimals in all 40 Camelyon cells, and Mahalanobis/ViM
reproduced for ConvNeXt, MobileNetV3, RegNetY and EffV2-S. For the 4 original anchors
(ResNet-18, ResNet-50, DenseNet-121, EffB3) Mahalanobis and/or ViM mismatched in 19/20 cells.

Diagnosis (same code, same feature file, node004):
- ResNet-18 seed 42 Mahalanobis AUROC: published 0.9558; re-runs 0.9619, 0.9519, 0.9558, 0.9579
  depending only on the CPU/BLAS thread setting. ViM: published 0.7148; re-runs 0.7111, 0.7337,
  0.7169, 0.7294.
- The Ledoit-Wolf covariance of the L2-normalised train features (302,436 x d) is fitted in
  float32 by `OODScorer.fit`; its condition number is ~1.7e10. The resulting precision matrix
  is not positive semi-definite (ResNet-18 seed 42: 12 negative eigenvalues; ResNet-50: 58), so
  squared Mahalanobis distances come out negative (down to -486 for ResNet-18 seed 42 in the
  `scores` job, -1500 for ResNet-50); `score_mahalanobis` clips them to 0.
- ViM with the default dimension (d = feat_dim - n_classes) keeps only the residual subspace of
  the n_classes = 2 smallest eigenvalues of that covariance, i.e. the numerically least
  determined directions.
The published Mahalanobis/ViM numbers of the 4 anchors are therefore one draw of a
thread-dependent computation, not a reproducible quantity.

## What is added (fixed before any of these numbers were read)

1. **Spread of the published code.** For the 4 anchors x seeds 42–46, the unchanged code path
   (`OODScorer.fit` + `score_mahalanobis`, `fit_vim` default dim + `vim_score`) is re-run 10
   times, each in a fresh process with OMP/MKL/OPENBLAS_NUM_THREADS cycling 1, 4, 8. Reported:
   mean, std, min, max AUROC, number of negative squared distances; Kendall W and every W REPRO
   number recomputed with each repeat (other 4 archs and other scores at their published values).
2. **Stable variant**, all 8 archs x 5 seeds:
   - Mahalanobis: L2-normalisation and sklearn `LedoitWolf` (automatic shrinkage) in float64;
     float64 quadratic form.
   - ViM: same `fit_vim`/`vim_score`, but d = smallest number of principal dimensions of the
     ViM covariance (around the ViM origin u, as in `fit_vim`) explaining >= 90% of its
     variance; float64.
   Each stable cell is computed 3–10 times over the same thread settings to show it is
   thread-invariant. Caveat known at writing time from two test cells: the 90% rule gives
   d = 2 (the covariance is taken around u, so the first direction carries most of the
   variance), so stable ViM keeps almost the whole residual space and is a materially
   different detector from the published ViM (which kept only 2 residual directions).
   Stable ViM is a sensitivity reading, not a replacement.
3. **Verdicts.** The rigor-pack stages are re-run unchanged on input trees where the anchor
   Mahalanobis/ViM values (CSV AUROC/FPR95, ranks file, per-sample score caches) are replaced
   by (a) each of the 10 old-code repeats and (b) the stable variant (all 8 archs). For every
   hypothesis that reads anchor Mahalanobis/ViM (H1, H2, H3, H4, H5, H7, H9, H16, H17; and H14
   through their families), the precommitted bar is applied to each tree.
   **Rule:** if the verdict is not identical in all 10 old repeats AND the stable variant, the
   hypothesis is reported as **INCONCLUSIVE (numerical instability)**, together with the
   verdicts that were obtained. Otherwise the common verdict stands.
   Not affected (do not read Mahalanobis/ViM): H6, H8, H10 (bars on MSP), H13, H15.
   H11(c) refits Mahalanobis on seed-42 indexed caches and inherits the same instability for
   the anchors; it is reported with that caveat. H12 (skin, MIDOG) uses published CSVs whose
   features are not re-fitted here; the same float32 path produced them, so their anchor
   Mahalanobis/ViM entries carry the same caveat.

Official W (0.694 / 0.791 / 0.857) is not replaced; the W spread is reported next to it.

Code: `scripts/rigor/numerical_stability.py`, `scripts/rigor/stability_readout.py`,
`scripts/rigor/verdict_reader.py`, `run_stability_variants.sh`.
Results: `outputs/reports/rigor_pack/numerical_stability/`.

## Results

Files: `outputs/reports/rigor_pack/numerical_stability/` (`old_runs.csv`, `old_spread.csv`,
`stable_vs_published.csv`, `w_per_variant.csv`, `w_spread.csv`, `verdicts_by_variant.csv`,
`verdicts_before_after.csv/.md`). Thread settings: repeat r uses (1, 4, 8)[r % 3] threads.

### Old code, 10 repeats (anchor 4 × 5 seeds × {Maha, ViM} = 40 cells)

- Within one thread setting the old code is deterministic (range 0); all spread comes from
  the BLAS thread count.
- Cells with range > 0.002 AUROC: Mahalanobis 15/20, ViM 18/20. Max range: Maha 0.360
  (EffB3 s46), ViM 0.612 (EffB3 s45). Mean per-cell std: Maha 0.032, ViM 0.035.
- Negative d² (clipped to 0) occurs in 13/20 Maha cells (up to 344k test samples for EffB3 s46).
- Max range per backbone (Maha / ViM): ResNet18 0.087 / 0.127, ResNet50 0.283 / 0.032,
  DenseNet121 0.000 / 0.070, EffB3 0.360 / 0.612.
- Share of runs within 2e-3 of the published AUROC: 1 thread 0.40, 4 threads 0.44,
  8 threads 0.94 (Maha 1.00, ViM 0.89).

| Worst cells | published | old min–max | negative d² |
|---|---|---|---|
| ResNet50 s43 Mahalanobis | 0.964 | 0.682–0.965 | 141k |
| EffB3 s46 Mahalanobis | 0.809 | 0.567–0.927 | 344k |
| EffB3 s45 ViM | 0.284 | 0.284–0.896 | – |

### Kendall W and REPRO lines

| Statistic | published | old 10 runs min–max | stable |
|---|---|---|---|
| W seed 42, 8 archs (REPRO 0.694) | 0.694 | 0.656–0.674 | 0.809 |
| Mean cross-arch W | 0.681 | 0.653–0.683 | 0.799 |
| Mean cross-seed W (8 archs) | 0.825 | 0.835–0.850 | 0.888 |
| Cross-seed W, EffB3 | 0.600 | 0.511–0.680 | 0.863 |
| Cross-seed W, ResNet18 | 0.800 | 0.800–0.931 | 0.931 |

No old-code run reproduces W = 0.694: the published seed-42 anchor ViM values come from the
ranks file, which matches no single thread setting. The transfer-cost REPRO (ResNet50 →
ConvNeXt, seed 42, nonfeature = 0.185) gives 0.1853 at 1 thread and 0.0646 at 4 and 8 threads;
the stable variant gives 0.000. MSP seed-42 REPRO does not involve Maha/ViM and is unchanged.

### Stable variant vs published (8 backbones × 5 seeds)

- Float64 Ledoit-Wolf Mahalanobis: no negative d², range 0 over repeats. Equal to published
  for the 4 non-anchor backbones; higher for ResNet18 (+0.040 mean), ResNet50 (+0.010) and
  EffB3 (+0.104); DenseNet121 unchanged.
- ViM at ≥90% variance: d = 2–4 for most backbones (MobileNet 24, RegNet 43 median). It is a
  materially different detector and is higher than published on every backbone (mean +0.04
  to +0.22). This drives most of the verdict changes below.

### Verdicts (rule above: final = verdict identical in all 10 old runs and in the stable variant)

| H | as run | old code, 10 runs | stable | final |
|---|---|---|---|---|
| H1 | mixed | arch separable 7/10, mixed 3/10 | mixed | INCONCLUSIVE (numerical instability) |
| H2 | all above chance | same 10/10 | same | all above chance |
| H3 | D>0; sampling not what moves W | same 10/10 | same | D>0; sampling not what moves W |
| H4 | arch effect detected | same 10/10 | same | arch effect detected |
| H5 | (a) every backbone and seed; (b) pass | same 10/10 | (a) "usually"; (b) fail | INCONCLUSIVE (numerical instability) |
| H7 | not AUROC-specific | same 10/10 | changes under FPR95 / AUPR-out | INCONCLUSIVE (numerical instability) |
| H9 | risk-level dependent; triage-score robust | same 10/10 | same | unchanged |
| H14 | descriptive (Holm within family) | same 10/10 | same | descriptive |
| H16 | 0.185 kept with percentile | same 10/10 (percentile 0.89–0.91) | percentile 0.982 → "worst case" wording | INCONCLUSIVE (numerical instability) |
| H17 | seeds agree more (D>0.10) | same 10/10 (D 0.14–0.17) | tie-robust (D = 0.070) | INCONCLUSIVE (numerical instability) |

H6, H8, H10, H11, H12, H13, H15 do not depend on anchor Maha/ViM and are not re-read here.

### Extra sensitivity readings (added 2026-10-04, after the spread above was seen)

Requested after the results above were known, so they are reported as extra readings only and
do not change the final column. `stable_maha_vim8` = stable Mahalanobis + original ViM at
8 threads (the draw closest to the published ViM). `stable_maha_novim` = stable Mahalanobis,
ViM removed from all 8 backbones (6 methods; W is not on the same scale as with 7 methods).

| H | stable Maha + ViM 8 threads | stable Maha, no ViM |
|---|---|---|
| H1 | mixed | mixed |
| H5 | (a) every backbone and seed; (b) pass | (a) every backbone and seed; (b) pass |
| H7 | not AUROC-specific | not AUROC-specific |
| H16 | 0.185 kept with percentile (0.911) | 0.185 kept with percentile (0.857) |
| H17 | seeds agree more (D = 0.139) | tie-robust (D = 0.099) |
| W seed 42 / mean cross-arch / mean cross-seed | 0.662 / 0.691 / 0.848 | 0.746 / 0.766 / 0.885 |

With the original ViM kept, only the Mahalanobis fix is applied and H1, H5, H7 and H16 read
as in the precommit run; H17 depends on ViM (D falls to 0.099, just under the 0.10 bar, once
ViM is dropped). The changes of H5, H7 and H16 under the stable variant come from the
90%-variance ViM, not from the Mahalanobis fix.

### Post-hoc: is the cross-arch transfer regret distinguishable from 0? (added 2026-10-05)

Not in the precommit (H16 has no CI); asked after the results above. `scripts/rigor/transfer_regret_ci.py`,
Camelyon17, nonfeature family, 280 cross-arch pairs. Mean regret with a 95% two-way cluster
bootstrap over archs and seeds (B = 10000); per pair, the AUROC loss on the target cell with a
patient-clustered DeLong SE, one-sided, Holm over the 280 pairs. Files:
`outputs/reports/rigor_pack/numerical_stability/transfer_regret_ci*.csv`.

| reading | mean regret [95% CI] | pairs > 0.05 | pairs Holm-sig. and > 0.05 | R50→ConvNeXt s42 |
|---|---|---|---|---|
| old code, 10 runs | 0.048–0.055, CI lower 0.014–0.019 | 35–41% | 15–22 | 0.185 or 0.065 |
| stable (Maha LW64 + ViM 90%) | 0.011 [0.000, 0.032] | 6.8% | 6 | 0.000 |
| extra: stable Maha + ViM 8 threads | 0.052 [0.019, 0.089] | 41% | 16 | 0.065 |
| extra: stable Maha, no ViM | 0.034 [0.000, 0.084] | 18% | 21 | 0.185 |

Under the stable variant the 90%-variance ViM is the best nonfeature score on 259/280 target
cells, so copying the source choice is almost always right; the 6 significant pairs all have
EfficientNet-V2-S (5) or DenseNet121 s42 (1) as source. The mean-regret CI includes 0 under
the stable variant and with ViM removed, and excludes 0 only when the original ViM is kept.
The "all" family behaves the same way (stable CI [0.000, 0.027]).
