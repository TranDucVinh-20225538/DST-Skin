# P2-c: two-channel table from the 18 balanced-fold retrains (R3 item 8, CPU, no new runs)

Camelyon isbi_patch2 v2 retrains, 3 CNNs x seeds 42-44 x 2 balanced slide folds. Each fold's backbone and scorer are fitted on the same 15 training slides (n_groups_fit = 15, K = 2). Every entry is the median over the 3 seeds. Delta_bb = AUROC(seen-slide id_val vs hospital 2) - AUROC(unseen-slide id_val vs hospital 2), within one model. For the logit scores (MSP, Energy, ReAct) Delta_bb is the backbone channel; for the feature scores (Mahalanobis, kNN, ViM) it is backbone + scorer-fit channels. Scorers: DST OODScorer (Track A definitions, float64). ViM: not numerically reproducible across BLAS thread counts (R3 item 1). [min, max] over seeds in the second table.

## Delta_bb (median over seeds), with ID accuracy

| arch | fold | d | acc seen | acc unseen | MSP | Energy | ReAct | Mahalanobis | kNN | ViM |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 0 | 2048 | 0.994 | 0.915 | +0.136 (bc) | +0.108 (bc) | +0.270 | +0.071 | +0.163 | +0.024 |
| resnet50 | 1 | 2048 | 0.997 | 0.658 | +0.195 | +0.200 | +0.237 | +0.114 | +0.232 | +0.068 |
| convnext_tiny | 0 | 768 | 0.996 | 0.967 | +0.135 | +0.129 | +0.187 | +0.104 | +0.129 | +0.086 |
| convnext_tiny | 1 | 768 | 0.998 | 0.707 | +0.251 | +0.247 | +0.133 | +0.215 | +0.253 | +0.149 |
| densenet121 | 0 | 1024 | 0.996 | 0.915 | +0.153 | +0.141 | +0.203 | +0.081 | +0.141 | +0.073 |
| densenet121 | 1 | 1024 | 0.997 | 0.655 | +0.248 | +0.239 | +0.244 | +0.125 | +0.251 | +0.107 |

(nc) near chance: seen and unseen AUROC both in [0.45, 0.55]. (bc) below chance: both < 0.5.

## AUROC seen / unseen (median over seeds) and Delta_bb [min, max] over seeds

| arch | fold | score | AUROC seen | AUROC unseen | Delta_bb | [min, max] |
|---|---|---|---|---|---|---|
| resnet50 | 0 | MSP | 0.352 | 0.229 | +0.136 | [+0.107, +0.178] |
| resnet50 | 0 | Energy | 0.300 | 0.218 | +0.108 | [+0.082, +0.118] |
| resnet50 | 0 | ReAct | 0.710 | 0.440 | +0.270 | [+0.248, +0.318] |
| resnet50 | 0 | Mahalanobis | 0.957 | 0.886 | +0.071 | [+0.032, +0.073] |
| resnet50 | 0 | kNN | 0.853 | 0.690 | +0.163 | [+0.133, +0.179] |
| resnet50 | 0 | ViM | 0.849 | 0.825 | +0.024 | [+0.001, +0.038] |
| resnet50 | 1 | MSP | 0.560 | 0.365 | +0.195 | [+0.136, +0.210] |
| resnet50 | 1 | Energy | 0.568 | 0.368 | +0.200 | [+0.128, +0.211] |
| resnet50 | 1 | ReAct | 0.667 | 0.430 | +0.237 | [+0.188, +0.246] |
| resnet50 | 1 | Mahalanobis | 0.984 | 0.872 | +0.114 | [+0.107, +0.150] |
| resnet50 | 1 | kNN | 0.928 | 0.696 | +0.232 | [+0.229, +0.255] |
| resnet50 | 1 | ViM | 0.757 | 0.689 | +0.068 | [+0.055, +0.082] |
| convnext_tiny | 0 | MSP | 0.831 | 0.696 | +0.135 | [+0.129, +0.148] |
| convnext_tiny | 0 | Energy | 0.849 | 0.725 | +0.129 | [+0.080, +0.134] |
| convnext_tiny | 0 | ReAct | 0.714 | 0.533 | +0.187 | [+0.173, +0.199] |
| convnext_tiny | 0 | Mahalanobis | 0.926 | 0.823 | +0.104 | [+0.096, +0.105] |
| convnext_tiny | 0 | kNN | 0.912 | 0.782 | +0.129 | [+0.126, +0.138] |
| convnext_tiny | 0 | ViM | 0.778 | 0.693 | +0.086 | [+0.086, +0.098] |
| convnext_tiny | 1 | MSP | 0.754 | 0.516 | +0.251 | [+0.227, +0.268] |
| convnext_tiny | 1 | Energy | 0.781 | 0.523 | +0.247 | [+0.205, +0.272] |
| convnext_tiny | 1 | ReAct | 0.528 | 0.403 | +0.133 | [+0.100, +0.160] |
| convnext_tiny | 1 | Mahalanobis | 0.896 | 0.681 | +0.215 | [+0.205, +0.270] |
| convnext_tiny | 1 | kNN | 0.843 | 0.590 | +0.253 | [+0.243, +0.291] |
| convnext_tiny | 1 | ViM | 0.772 | 0.621 | +0.149 | [+0.139, +0.151] |
| densenet121 | 0 | MSP | 0.722 | 0.570 | +0.153 | [+0.150, +0.157] |
| densenet121 | 0 | Energy | 0.698 | 0.557 | +0.141 | [+0.141, +0.157] |
| densenet121 | 0 | ReAct | 0.850 | 0.647 | +0.203 | [+0.167, +0.216] |
| densenet121 | 0 | Mahalanobis | 0.980 | 0.899 | +0.081 | [+0.048, +0.096] |
| densenet121 | 0 | kNN | 0.919 | 0.756 | +0.141 | [+0.122, +0.163] |
| densenet121 | 0 | ViM | 0.748 | 0.696 | +0.073 | [+0.043, +0.090] |
| densenet121 | 1 | MSP | 0.598 | 0.348 | +0.248 | [+0.154, +0.250] |
| densenet121 | 1 | Energy | 0.586 | 0.347 | +0.239 | [+0.139, +0.252] |
| densenet121 | 1 | ReAct | 0.657 | 0.393 | +0.244 | [+0.203, +0.264] |
| densenet121 | 1 | Mahalanobis | 0.992 | 0.867 | +0.125 | [+0.122, +0.177] |
| densenet121 | 1 | kNN | 0.971 | 0.730 | +0.251 | [+0.191, +0.272] |
| densenet121 | 1 | ViM | 0.806 | 0.687 | +0.107 | [+0.061, +0.119] |

## Old single run (logit_retrain_slide_disjoint, seed 42, fold 0) beside the new fold-0 median

| arch | score | old Delta_bb | new fold-0 median | old acc unseen | new fold-0 acc unseen |
|---|---|---|---|---|---|
| resnet50 | MSP | +0.240 | +0.136 | 0.617 | 0.915 |
| resnet50 | Energy | +0.238 | +0.108 | 0.617 | 0.915 |
| convnext_tiny | MSP | +0.259 | +0.135 | 0.667 | 0.967 |
| convnext_tiny | Energy | +0.280 | +0.129 | 0.667 | 0.967 |
| densenet121 | MSP | +0.215 | +0.153 | 0.617 | 0.915 |
| densenet121 | Energy | +0.214 | +0.141 | 0.617 | 0.915 |

The old run scored MSP / Energy only; the other four scores have no old value. Old fold 0 is a different slide partition (unbalanced; 97,099 training patches vs 172,943 in balanced fold 0), so the old / new rows compare protocols, not the same slides. ELogitNorm is in two_channel_balanced_table.csv (not one of the six table scores).
