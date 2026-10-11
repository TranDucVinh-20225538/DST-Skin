# T1 item D — label-free overlap fingerprint T

**Verdict: NO-GO** (D-i fail, D-ii fail, D-iii pass). Rule: D-i or D-iii fail → NO-GO.

| sub-criterion | pre-registered threshold | computed | status |
|---|---|---|---|
| D-i (synthetic TEST, normalised T, frozen τ) | sensitivity ≥ 0.90 and specificity ≥ 0.90 on material TEST configs | sens **0.459**, spec 0.940 (τ = 1.490; 271 material TEST configs) | **fail** |
| D-ii (adversarial search) | no config with Δ_full ≥ 0.02 and AUC(T) < 0.8 | **996 falsifiers** in 4,280 evaluations (143 of 3,000 random, 853 of 1,280 CMA-ES) | **fail** |
| D-iii (real, Camelyon) | AUC(T_seen vs T_unseen) ≥ 0.90 in ≥ 11/13 backbones | **12/13** (dinov2_vitl14 0.833; all others ≥ 0.9997) | pass |

GPU: 2.82 GPU-h measured (D1 + D2 + D3/D4 raw seconds). Tests: 6 rows in `tests.csv` (1 post-hoc: D5). Deviations: 12
(`DEVIATIONS.md`, DD-1 … DD-12).

## D1 — synthetic calibration (TRAIN d ∈ {32, 512, 2048} → TEST d ∈ {128, 1024})
- τ maximises Youden's J on every replicate of the 600 TRAIN configs (ω = 1 vs ω = 0): train J 0.828 (sens 0.980, spec 0.848).
- On the material TEST configs, sensitivity by overlap fraction: ω = 0.25 → **0.169**, ω = 0.5 → **0.215**, ω = 1 → 0.994.
  T detects full overlap but misses partial overlap in most replicates; the pooled sensitivity over ω ≥ 0.25 is 0.459.
- Specificity 0.940; per-config AUC(T) ω = 0 vs 1: median 1.0, minimum 0.0 over material TEST configs.
- Report-only: the naive z test of T at ρ = 0 (|T| > 1.96) has size **0.188** (nominal 0.05); raw-feature T: sens 0.479, spec 0.912.

## D2 — falsifier search
996 configurations have a material leak (Δ_full ≥ 0.02) while AUC(T) < 0.8; the ten worst (`d2_worst10.csv`) are all
CMA-ES points with AUC(T_norm) 0.31–0.37 at d ≈ 500–2000, ρ ≈ 0.11–0.21, G ≈ 225–500, power-law spectra (α ≈ 1.3–1.5)
and Δ_full 0.03–0.12, i.e. regimes where the leak exists and T is blind or inverted.

## D3 — real features (Camelyon, 13 backbones, cells × folds × 50 subsamples)
Per-backbone AUC(T_seen vs T_unseen): conch_v1_5 1.0, convnext_tiny 1.0, densenet121 1.0, dinov2_vitb14 0.9997,
**dinov2_vitl14 0.833**, effb3 1.0, efficientnet_v2_s 1.0, mobilenet_v3_large 1.0, regnet_y_3_2gf 1.0, resnet18 1.0,
resnet50 1.0, uni 1.0, virchow2 1.0 → 12/13 ≥ 0.90. On the R3 Camelyon folds (whole-slide overlap, ω = 1) T separates seen
from unseen eval sets, consistent with the ω = 1 synthetic sensitivity; this does not address partial overlap (D-i).

Report-only (LOBO): T_unseen as a predictor of Δ — Mahalanobis: Spearman −0.08, mean D vs ICC −0.013 [−0.022, −0.003]
(T worse than ICC); kNN-50: Spearman 0.12, mean D −0.0001 [−0.004, 0.003].

## D4 — null (random half-split within the fit set)
316 cell-folds: median mean-T_null −0.064 (range −3.68 to 2.40); 6.0 % of cell-folds have |mean T_null| > 1.96.
The permuted-label check holds by construction (T is label-free; identical in 316/316).

## D5 (post-hoc)
Cluster-bootstrap SE / naive SE of the mean NN² difference: median **3.11** (IQR 1.46–3.96) over 316 cell-folds — the
naive SE of T understates its sampling variability by about 3× on real features; the z-scale of T is not calibrated.

## Reading
T flags complete group overlap (ω = 1) on synthetic and real features, but fails the pre-registered partial-overlap
sensitivity (D-i) and has a large adversarial blind region (D-ii). The D verdict is NO-GO.

Files: `d1_configs.csv`, `d1_tau.json`, `d2_search.csv`, `d2_worst10.csv`, `d3_real.csv`, `d4_null.csv`,
`d5_se_posthoc.csv`, `tests.csv`, `verdict.json`, `DEVIATIONS.md`.
