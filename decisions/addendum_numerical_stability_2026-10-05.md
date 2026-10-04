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

(filled in after the runs, in a separate commit)
