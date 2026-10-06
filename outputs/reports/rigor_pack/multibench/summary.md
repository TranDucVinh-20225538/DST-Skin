# Multibench group-leakage check: summary

Precommit `decisions/precommit_group_leakage_multibench_2026-10-06.md`. (A): Δ = disjoint − same-group
2-fold scorer-fit AUROC (negative = inflation from group sharing), seed 42. (B): within-model gap
= AUROC(seen-group ID) − AUROC(unseen-group ID), fold mean, mean ± SD over seeds 42/43. (C): winner
of the 7 scores under standard vs group-disjoint protocol, Kendall tau-b.

## iwildcam

| arch | ΔMaha | ΔkNN | ΔViM | ΔReAct | gap MSP | gap Energy | acc seen / unseen | winner std → disjoint | tau |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 | -0.222 | -0.224 | -0.068 | -0.001 | +0.134 ± 0.013 | +0.157 ± 0.017 | 0.714 / 0.563 | kNN → Energy | -0.14 |
| resnet50 | -0.304 | -0.295 | -0.168 | -0.003 | +0.143 ± 0.004 | +0.171 ± 0.017 | 0.729 / 0.582 | kNN → Energy | -0.05 |
| densenet121 | -0.356 | -0.273 | -0.080 | +0.000 | +0.134 ± 0.015 | +0.159 ± 0.026 | 0.726 / 0.581 | kNN → Energy | 0.05 |
| convnext_tiny | -0.162 | -0.186 | -0.168 | +0.000 | +0.149 ± 0.030 | +0.163 ± 0.030 | 0.773 / 0.662 | kNN → ELogitNorm | -0.05 |
| mobilenet_v3_large | -0.283 | -0.314 | -0.114 | -0.002 | — | — | — | kNN → Energy | -0.33 |
| regnet_y_3_2gf | -0.297 | -0.298 | -0.142 | -0.002 | — | — | — | kNN → Energy | 0.05 |
| effb3 | -0.213 | -0.214 | -0.155 | -0.000 | — | — | — | kNN → ELogitNorm | -0.43 |
| efficientnet_v2_s | -0.279 | -0.176 | -0.140 | -0.001 | — | — | — | kNN → Energy | -0.05 |

Same-protocol winner on retrained models (unseen-group ID, 16 runs): ELogitNorm 5, Mahalanobis 4, ViM 4, ReAct 2, kNN 1

Standard-protocol AUROC (7 scores):

| arch | MSP | Energy | ELogitNorm | ViM | ReAct | Mahalanobis | kNN |
|---|---|---|---|---|---|---|---|
| resnet18 | 0.605 | 0.628 | 0.491 | 0.637 | 0.606 | 0.674 | 0.773 |
| resnet50 | 0.603 | 0.612 | 0.493 | 0.712 | 0.590 | 0.799 | 0.806 |
| densenet121 | 0.575 | 0.604 | 0.425 | 0.648 | 0.590 | 0.698 | 0.781 |
| convnext_tiny | 0.601 | 0.619 | 0.628 | 0.687 | 0.395 | 0.655 | 0.704 |
| mobilenet_v3_large | 0.558 | 0.581 | 0.476 | 0.635 | 0.573 | 0.709 | 0.770 |
| regnet_y_3_2gf | 0.580 | 0.612 | 0.445 | 0.698 | 0.593 | 0.748 | 0.811 |
| effb3 | 0.577 | 0.595 | 0.564 | 0.650 | 0.558 | 0.692 | 0.710 |
| efficientnet_v2_s | 0.650 | 0.689 | 0.404 | 0.734 | 0.666 | 0.759 | 0.771 |

## rxrx1

| arch | ΔMaha | ΔkNN | ΔViM | ΔReAct | gap MSP | gap Energy | acc seen / unseen | winner std → disjoint | tau |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 | -0.077 | -0.066 | -0.002 | -0.000 | -0.045 ± 0.006 | +0.001 ± 0.004 | 0.075 / 0.057 | ViM → ViM | 0.90 |
| resnet50 | -0.091 | -0.055 | -0.022 | -0.000 | -0.035 ± 0.004 | +0.003 ± 0.004 | 0.105 / 0.074 | ViM → ReAct | 0.90 |
| densenet121 | -0.095 | -0.074 | +0.000 | -0.000 | -0.039 ± 0.005 | +0.020 ± 0.004 | 0.076 / 0.059 | kNN → ReAct | 0.71 |
| convnext_tiny | -0.045 | -0.060 | -0.002 | -0.000 | -0.010 ± 0.012 | -0.020 ± 0.002 | 0.078 / 0.058 | ViM → ViM | 0.62 |

Same-protocol winner on retrained models (unseen-group ID, 16 runs): ReAct 6, ViM 6, Energy 2, ELogitNorm 2

Standard-protocol AUROC (7 scores):

| arch | MSP | Energy | ELogitNorm | ViM | ReAct | Mahalanobis | kNN |
|---|---|---|---|---|---|---|---|
| resnet18 | 0.440 | 0.520 | 0.439 | 0.561 | 0.542 | 0.512 | 0.531 |
| resnet50 | 0.443 | 0.537 | 0.433 | 0.562 | 0.559 | 0.514 | 0.529 |
| densenet121 | 0.448 | 0.510 | 0.453 | 0.510 | 0.526 | 0.509 | 0.540 |
| convnext_tiny | 0.477 | 0.463 | 0.466 | 0.584 | 0.452 | 0.453 | 0.497 |

