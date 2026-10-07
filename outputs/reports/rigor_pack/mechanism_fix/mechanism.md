# Mechanism (M1-M4)

Precommit `decisions/precommit_leak_mechanism_fix_2026-10-06.md`. Δ = AUROC_same − AUROC_disjoint
(2-fold fit protocol, seed 42); reduction = 1 − effect_ablated / effect_original.

## M1 stain / colour ablation

Cells with |original effect| < 0.01 are excluded from the median (nothing to reduce; mostly ReAct) and counted.

| dataset | variant | median reduction (cells) | all-cells median | excluded | published ID acc original → ablated |
|---|---|---|---|---|---|
| camelyon | gray | +77% (18) | +86% | 4 | 0.996 → 0.556 |
| camelyon | macenko | +80% (18) | +82% | 4 | 0.996 → 0.601 |
| iwildcam | colour | +7% (20) | +7% | 4 | 0.765 → 0.716 |
| iwildcam | gray | +10% (20) | +13% | 4 | 0.765 → 0.589 |
| rxrx1 | colour | +57% (14) | +57% | 10 | 0.187 → 0.004 |
| rxrx1 | gray | +55% (14) | +42% | 10 | 0.187 → 0.005 |

Verdicts: camelyon: group appearance is the main driver; iwildcam: not explained by colour / stain; rxrx1: group appearance is the main driver

Per effect (median reduction over archs, counted cells):

| dataset | variant | ΔMahalanobis | ΔkNN | ΔViM | ΔReAct | gap MSP | gap Energy |
|---|---|---|---|---|---|---|---|
| camelyon | gray | +48% | -11% | +78% | — | +110% | +110% |
| camelyon | macenko | +51% | +21% | +60% | — | +139% | +142% |
| iwildcam | colour | +2% | -12% | +6% | — | +19% | +18% |
| iwildcam | gray | +4% | -27% | -9% | — | +22% | +28% |
| rxrx1 | colour | +67% | +41% | +27% | — | +164% | +80% |
| rxrx1 | gray | +67% | +32% | +36% | — | +127% | +45% |

## M2 group decodability / M3 same-group neighbours / centroid distance

| dataset | arch | ΔMaha | ΔkNN | ΔViM | ΔReAct | M2 bal. acc (chance) | M3 frac (chance) | centroid dist |
|---|---|---|---|---|---|---|---|---|
| camelyon | convnext_tiny | +0.059 | +0.054 | +0.074 | +0.002 | 0.79 (0.033) | 0.76 (0.082) | 0.0018 |
| camelyon | densenet121 | +0.113 | +0.093 | +0.078 | +0.000 | 0.77 (0.033) | 0.69 (0.082) | 0.0005 |
| camelyon | effb3 | +0.140 | +0.136 | +0.018 | +0.005 | 0.72 (0.033) | 0.58 (0.082) | 0.0036 |
| camelyon | efficientnet_v2_s | +0.083 | +0.086 | +0.086 | +0.002 | 0.74 (0.033) | 0.67 (0.082) | 0.0012 |
| camelyon | mobilenet_v3_large | +0.099 | +0.091 | +0.055 | -0.001 | 0.72 (0.033) | 0.73 (0.082) | 0.0048 |
| camelyon | regnet_y_3_2gf | +0.149 | +0.122 | +0.080 | -0.001 | 0.77 (0.033) | 0.78 (0.082) | 0.0015 |
| camelyon | resnet18 | +0.168 | +0.136 | +0.108 | +0.002 | 0.71 (0.033) | 0.69 (0.082) | 0.0002 |
| camelyon | resnet50 | +0.216 | +0.203 | +0.074 | -0.000 | 0.72 (0.033) | 0.65 (0.082) | 0.0003 |
| iwildcam | convnext_tiny | +0.162 | +0.186 | +0.168 | -0.000 | 0.85 (0.005) | 0.76 (0.013) | 0.2591 |
| iwildcam | densenet121 | +0.356 | +0.273 | +0.080 | -0.000 | 0.88 (0.005) | 0.73 (0.013) | 0.1105 |
| iwildcam | effb3 | +0.213 | +0.214 | +0.155 | +0.000 | 0.80 (0.005) | 0.71 (0.013) | 0.1412 |
| iwildcam | efficientnet_v2_s | +0.279 | +0.176 | +0.140 | +0.001 | 0.81 (0.005) | 0.68 (0.013) | 0.2007 |
| iwildcam | mobilenet_v3_large | +0.283 | +0.314 | +0.114 | +0.002 | 0.83 (0.005) | 0.76 (0.013) | 0.1574 |
| iwildcam | regnet_y_3_2gf | +0.297 | +0.298 | +0.142 | +0.002 | 0.84 (0.005) | 0.73 (0.013) | 0.1047 |
| iwildcam | resnet18 | +0.222 | +0.224 | +0.068 | +0.001 | 0.81 (0.005) | 0.70 (0.013) | 0.0869 |
| iwildcam | resnet50 | +0.304 | +0.295 | +0.168 | +0.003 | 0.84 (0.005) | 0.67 (0.013) | 0.0784 |
| rxrx1 | convnext_tiny | +0.045 | +0.060 | +0.002 | +0.000 | 0.70 (0.030) | 0.23 (0.030) | 0.0242 |
| rxrx1 | densenet121 | +0.095 | +0.074 | -0.000 | +0.000 | 0.75 (0.030) | 0.23 (0.030) | 0.0015 |
| rxrx1 | resnet18 | +0.077 | +0.066 | +0.002 | +0.000 | 0.61 (0.030) | 0.22 (0.030) | 0.0014 |
| rxrx1 | resnet50 | +0.091 | +0.055 | +0.022 | +0.000 | 0.73 (0.030) | 0.19 (0.030) | 0.0011 |

Spearman(M3 excess, Δ) per fit score over archs (descriptive):

- camelyon: Mahalanobis -0.26, kNN -0.52, ViM +0.33
- iwildcam: Mahalanobis -0.12, kNN +0.31, ViM -0.07
- rxrx1: Mahalanobis -0.40, kNN +0.40, ViM -0.40

## M4 instance vs group

| dataset | fit scores median R (n, excluded) | verdict | MSP/Energy median R (n, excluded) | verdict |
|---|---|---|---|---|
| camelyon | 0.93 (25, 7) | group-level effect | 0.97 (6, 0) | group-level effect |
| iwildcam | 0.58 (32, 0) | group-level effect | 0.58 (8, 0) | group-level effect |
| rxrx1 | 0.61 (9, 7) | group-level effect | 1.17 (6, 2) | group-level effect |

## Main-driver sentence

- camelyon: M1 → group appearance is the main driver; M4 → fit scores group-level effect, logit scores group-level effect.
- iwildcam: M1 → not explained by colour / stain; M4 → fit scores group-level effect, logit scores group-level effect.
- rxrx1: M1 → group appearance is the main driver; M4 → fit scores group-level effect, logit scores group-level effect.
