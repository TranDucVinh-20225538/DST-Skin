# R3 item 2: coverage of the paper_2fold jackknife CI on designs read from the real splits

Commit: 0bdd906

Verdict: 1 cell(s) with jackknife coverage < 0.93: reported, stop (no method change this round).

Design: crossfit-ood v3 coverage_study_paper2fold.py (ICC grid, decision rule, defaults unchanged) on the real per-group train sizes, ID-eval sizes and OOD sizes of every item-1 dataset (scripts/r3/item2_coverage_real.py). Scorers: Track A definitions (mahalanobis_l2, knn_mean_cosine); seen = strict only.

Datasets per cell: min 100 (target 100).

## Real designs

```
camelyon {"n_train_groups": 30, "n_train": 10080, "n_groups_with_eval": 30, "n_eval": 1117, "n_ood_groups": 2835, "n_ood": 2835, "train_size_min_med_max": [48, 220.0, 1838], "scale": [0.03333333333333333, 0.03333333333333333, 0.03333333333333333]} {"real_train": 302436, "real_groups": 30, "real_eval": 33560, "real_eval_groups": 30, "real_eval_not_in_train": 0, "real_ood": 85054, "source": "camelyon_resnet50_s42"}
breakhis {"n_train_groups": 56, "n_train": 3661, "n_groups_with_eval": 56, "n_eval": 1017, "n_ood_groups": 1013, "n_ood": 1013, "train_size_min_med_max": [24, 59.0, 152], "scale": [1.0, 1.0, 1.0]} {"real_train": 3661, "real_groups": 56, "real_eval": 1017, "real_eval_groups": 56, "real_eval_not_in_train": 0, "real_ood": 1013, "source": "breakhis_resnet50_s42_std_r0"}
dermamnist {"n_train_groups": 5678, "n_train": 7007, "n_groups_with_eval": 641, "n_eval": 693, "n_ood_groups": 2298, "n_ood": 2298, "train_size_min_med_max": [1, 1.0, 6], "scale": [1.0, 1.0, 1.0]} {"real_train": 7007, "real_groups": 5678, "real_eval": 693, "real_eval_groups": 641, "real_eval_not_in_train": 0, "real_ood": 2298, "source": "dermamnist_resnet50_s42_std"}
isic2019 {"n_train_groups": 11041, "n_train": 17884, "n_groups_with_eval": 2201, "n_eval": 2904, "n_ood_groups": 492, "n_ood": 492, "train_size_min_med_max": [1, 1.0, 24], "scale": [1.0, 1.0, 1.0]} {"real_train": 17884, "real_groups": 11041, "real_eval": 2904, "real_eval_groups": 2201, "real_eval_not_in_train": 117, "real_ood": 492, "source": "isic2019_resnet50_s42_std"}
kermany {"n_train_groups": 4240, "n_train": 33932, "n_groups_with_eval": 435, "n_eval": 659, "n_ood_groups": 1108, "n_ood": 1108, "train_size_min_med_max": [1, 3.0, 360], "scale": [0.5, 1.0, 0.125]} {"real_train": 67381, "real_groups": 4240, "real_eval": 659, "real_eval_groups": 435, "real_eval_not_in_train": 5, "real_ood": 8866, "source": "kermany_resnet50_s42_std"}
```

## Pre-registered decision (jackknife, 0.93)

Cells: 49; jackknife coverage min = 0.92; cells < 0.93: 1.

- camelyon / knn_mean_cosine / moderate: 0.92 (n = 100)

## Coverage per cell (d = 128, v3 default)

| design | scorer | level | true Δ | n | jackknife coverage ± MC SE | old bootstrap coverage ± MC SE | jackknife width | P(lo > 0) | P(lo > 0.02) |
|---|---|---|---|---|---|---|---|---|---|
| camelyon | knn_mean_cosine | null | +0.0024 | 100 | 0.98 ± 0.01 | 0.82 ± 0.04 | 0.0197 | 0.02 | 0.00 |
| camelyon | knn_mean_cosine | d01 | +0.0093 | 100 | 0.98 ± 0.01 | 0.84 ± 0.04 | 0.0234 | 0.31 | 0.00 |
| camelyon | knn_mean_cosine | d03 | +0.0249 | 100 | 0.97 ± 0.02 | 0.88 ± 0.03 | 0.0344 | 0.90 | 0.00 |
| camelyon | knn_mean_cosine | d05 | +0.0476 | 100 | 0.94 ± 0.02 | 0.91 ± 0.03 | 0.0494 | 1.00 | 0.68 |
| camelyon | knn_mean_cosine | moderate | +0.1416 | 100 | 0.92 ± 0.03 | 0.91 ± 0.03 | 0.0852 | 1.00 | 1.00 |
| camelyon | mahalanobis_l2 | null | +0.0022 | 100 | 0.93 ± 0.03 | 0.75 ± 0.04 | 0.0170 | 0.01 | 0.00 |
| camelyon | mahalanobis_l2 | d01 | +0.0093 | 100 | 0.96 ± 0.02 | 0.85 ± 0.04 | 0.0203 | 0.55 | 0.00 |
| camelyon | mahalanobis_l2 | d03 | +0.0245 | 100 | 0.96 ± 0.02 | 0.92 ± 0.03 | 0.0297 | 1.00 | 0.01 |
| camelyon | mahalanobis_l2 | d05 | +0.0448 | 100 | 0.95 ± 0.02 | 0.90 ± 0.03 | 0.0402 | 1.00 | 0.80 |
| camelyon | mahalanobis_l2 | moderate | +0.1445 | 100 | 0.93 ± 0.03 | 0.91 ± 0.03 | 0.0613 | 1.00 | 1.00 |
| breakhis | knn_mean_cosine | null | +0.0006 | 100 | 0.99 ± 0.01 | 0.90 ± 0.03 | 0.0105 | 0.02 | 0.00 |
| breakhis | knn_mean_cosine | d01 | +0.0080 | 100 | 0.99 ± 0.01 | 0.90 ± 0.03 | 0.0120 | 0.82 | 0.00 |
| breakhis | knn_mean_cosine | d03 | +0.0243 | 100 | 0.98 ± 0.01 | 0.94 ± 0.02 | 0.0153 | 1.00 | 0.16 |
| breakhis | knn_mean_cosine | d05 | +0.0485 | 100 | 0.98 ± 0.01 | 0.91 ± 0.03 | 0.0198 | 1.00 | 1.00 |
| breakhis | knn_mean_cosine | moderate | +0.1290 | 100 | 0.98 ± 0.01 | 0.93 ± 0.03 | 0.0359 | 1.00 | 1.00 |
| breakhis | mahalanobis_l2 | null | +0.0007 | 100 | 1.00 ± 0.00 | 0.88 ± 0.03 | 0.0062 | 0.00 | 0.00 |
| breakhis | mahalanobis_l2 | d01 | +0.0079 | 100 | 0.99 ± 0.01 | 0.88 ± 0.03 | 0.0088 | 0.99 | 0.00 |
| breakhis | mahalanobis_l2 | d03 | +0.0238 | 100 | 0.99 ± 0.01 | 0.90 ± 0.03 | 0.0137 | 1.00 | 0.09 |
| breakhis | mahalanobis_l2 | d05 | +0.0444 | 100 | 0.99 ± 0.01 | 0.93 ± 0.03 | 0.0189 | 1.00 | 1.00 |
| breakhis | mahalanobis_l2 | moderate | +0.1367 | 100 | 0.99 ± 0.01 | 0.98 ± 0.01 | 0.0370 | 1.00 | 1.00 |
| dermamnist | knn_mean_cosine | null | +0.0001 | 100 | 0.98 ± 0.01 | 0.96 ± 0.02 | 0.0098 | 0.00 | 0.00 |
| dermamnist | knn_mean_cosine | d01 | +0.0079 | 100 | 0.99 ± 0.01 | 0.95 ± 0.02 | 0.0098 | 0.90 | 0.00 |
| dermamnist | knn_mean_cosine | d03 | +0.0297 | 100 | 0.99 ± 0.01 | 0.98 ± 0.01 | 0.0105 | 1.00 | 0.99 |
| dermamnist | knn_mean_cosine | d05 | +0.0496 | 100 | 0.99 ± 0.01 | 0.99 ± 0.01 | 0.0118 | 1.00 | 1.00 |
| dermamnist | knn_mean_cosine | max | +0.0887 | 100 | 1.00 ± 0.00 | 0.98 ± 0.01 | 0.0156 | 1.00 | 1.00 |
| dermamnist | mahalanobis_l2 | null | +0.0001 | 100 | 1.00 ± 0.00 | 0.94 ± 0.02 | 0.0039 | 0.01 | 0.00 |
| dermamnist | mahalanobis_l2 | d01 | +0.0093 | 100 | 0.98 ± 0.01 | 0.93 ± 0.03 | 0.0058 | 1.00 | 0.00 |
| dermamnist | mahalanobis_l2 | d03 | +0.0298 | 100 | 0.99 ± 0.01 | 0.96 ± 0.02 | 0.0097 | 1.00 | 1.00 |
| dermamnist | mahalanobis_l2 | max | +0.0364 | 100 | 1.00 ± 0.00 | 0.95 ± 0.02 | 0.0109 | 1.00 | 1.00 |
| isic2019 | knn_mean_cosine | null | +0.0000 | 100 | 0.99 ± 0.01 | 0.97 ± 0.02 | 0.0048 | 0.01 | 0.00 |
| isic2019 | knn_mean_cosine | d01 | +0.0072 | 100 | 0.99 ± 0.01 | 0.92 ± 0.03 | 0.0050 | 1.00 | 0.00 |
| isic2019 | knn_mean_cosine | d03 | +0.0276 | 100 | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.0067 | 1.00 | 1.00 |
| isic2019 | knn_mean_cosine | d05 | +0.0477 | 100 | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.0090 | 1.00 | 1.00 |
| isic2019 | knn_mean_cosine | moderate | +0.1483 | 100 | 0.96 ± 0.02 | 0.96 ± 0.02 | 0.0230 | 1.00 | 1.00 |
| isic2019 | mahalanobis_l2 | null | +0.0000 | 100 | 1.00 ± 0.00 | 0.98 ± 0.01 | 0.0024 | 0.00 | 0.00 |
| isic2019 | mahalanobis_l2 | d01 | +0.0092 | 100 | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.0040 | 1.00 | 0.00 |
| isic2019 | mahalanobis_l2 | d03 | +0.0294 | 100 | 1.00 ± 0.00 | 0.96 ± 0.02 | 0.0078 | 1.00 | 1.00 |
| isic2019 | mahalanobis_l2 | d05 | +0.0493 | 100 | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.0116 | 1.00 | 1.00 |
| isic2019 | mahalanobis_l2 | max | +0.0563 | 100 | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.0129 | 1.00 | 1.00 |
| kermany | knn_mean_cosine | null | -0.0000 | 100 | 1.00 ± 0.00 | 0.94 ± 0.02 | 0.0092 | 0.00 | 0.00 |
| kermany | knn_mean_cosine | d01 | +0.0066 | 100 | 0.97 ± 0.02 | 0.95 ± 0.02 | 0.0102 | 0.79 | 0.00 |
| kermany | knn_mean_cosine | d03 | +0.0219 | 100 | 1.00 ± 0.00 | 0.98 ± 0.01 | 0.0144 | 1.00 | 0.01 |
| kermany | knn_mean_cosine | d05 | +0.0450 | 100 | 1.00 ± 0.00 | 0.99 ± 0.01 | 0.0219 | 1.00 | 1.00 |
| kermany | knn_mean_cosine | moderate | +0.1483 | 100 | 0.99 ± 0.01 | 0.98 ± 0.01 | 0.0466 | 1.00 | 1.00 |
| kermany | mahalanobis_l2 | null | +0.0001 | 100 | 0.98 ± 0.01 | 0.96 ± 0.02 | 0.0045 | 0.00 | 0.00 |
| kermany | mahalanobis_l2 | d01 | +0.0077 | 100 | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.0087 | 0.99 | 0.00 |
| kermany | mahalanobis_l2 | d03 | +0.0263 | 100 | 0.99 ± 0.01 | 0.99 ± 0.01 | 0.0171 | 1.00 | 0.17 |
| kermany | mahalanobis_l2 | d05 | +0.0460 | 100 | 0.99 ± 0.01 | 0.99 ± 0.01 | 0.0251 | 1.00 | 1.00 |
| kermany | mahalanobis_l2 | moderate | +0.1489 | 100 | 0.97 ± 0.02 | 0.96 ± 0.02 | 0.0594 | 1.00 | 1.00 |

## Power at true Δ ≈ 0.03 (level d03; jackknife)

| design | scorer | true Δ | P(lo > 0) | P(lo > 0.02) | n |
|---|---|---|---|---|---|
| camelyon | knn_mean_cosine | +0.0249 | 0.90 | 0.00 | 100 |
| camelyon | mahalanobis_l2 | +0.0245 | 1.00 | 0.01 | 100 |
| breakhis | knn_mean_cosine | +0.0243 | 1.00 | 0.16 | 100 |
| breakhis | mahalanobis_l2 | +0.0238 | 1.00 | 0.09 | 100 |
| dermamnist | knn_mean_cosine | +0.0297 | 1.00 | 0.99 | 100 |
| dermamnist | mahalanobis_l2 | +0.0298 | 1.00 | 1.00 | 100 |
| isic2019 | knn_mean_cosine | +0.0276 | 1.00 | 1.00 | 100 |
| isic2019 | mahalanobis_l2 | +0.0294 | 1.00 | 1.00 | 100 |
| kermany | knn_mean_cosine | +0.0219 | 1.00 | 0.01 | 100 |
| kermany | mahalanobis_l2 | +0.0263 | 1.00 | 0.17 | 100 |

## Real dimension check (Camelyon design)

ICC levels from the d = 128 calibration; true Δ re-estimated at each d.

| scorer | level | d | true Δ | n | jackknife coverage ± MC SE | P(lo > 0) |
|---|---|---|---|---|---|---|
| knn_mean_cosine | null | 128 | +0.0024 | 100 | 0.98 ± 0.01 | 0.02 |
| knn_mean_cosine | d01 | 128 | +0.0093 | 100 | 0.98 ± 0.01 | 0.31 |
| knn_mean_cosine | d03 | 128 | +0.0249 | 100 | 0.97 ± 0.02 | 0.90 |
| knn_mean_cosine | d05 | 128 | +0.0476 | 100 | 0.94 ± 0.02 | 1.00 |
| knn_mean_cosine | moderate | 128 | +0.1416 | 100 | 0.92 ± 0.03 | 1.00 |
| mahalanobis_l2 | null | 128 | +0.0022 | 100 | 0.93 ± 0.03 | 0.01 |
| mahalanobis_l2 | d01 | 128 | +0.0093 | 100 | 0.96 ± 0.02 | 0.55 |
| mahalanobis_l2 | d03 | 128 | +0.0245 | 100 | 0.96 ± 0.02 | 1.00 |
| mahalanobis_l2 | d05 | 128 | +0.0448 | 100 | 0.95 ± 0.02 | 1.00 |
| mahalanobis_l2 | moderate | 128 | +0.1445 | 100 | 0.93 ± 0.03 | 1.00 |
| knn_mean_cosine | null | 768 | +0.0009 | 100 | 0.98 ± 0.01 | 0.02 |
| knn_mean_cosine | d01 | 768 | +0.0207 | 100 | 0.96 ± 0.02 | 1.00 |
| knn_mean_cosine | d03 | 768 | +0.0770 | 100 | 0.93 ± 0.03 | 1.00 |
| knn_mean_cosine | d05 | 768 | +0.1692 | 100 | 0.96 ± 0.02 | 1.00 |
| knn_mean_cosine | moderate | 768 | +0.4017 | 100 | 0.98 ± 0.01 | 1.00 |
| mahalanobis_l2 | null | 768 | +0.0009 | 100 | 0.93 ± 0.03 | 0.01 |
| mahalanobis_l2 | d01 | 768 | +0.0191 | 100 | 0.95 ± 0.02 | 1.00 |
| mahalanobis_l2 | d03 | 768 | +0.0758 | 100 | 0.97 ± 0.02 | 1.00 |
| mahalanobis_l2 | d05 | 768 | +0.1544 | 100 | 0.97 ± 0.02 | 1.00 |
| mahalanobis_l2 | moderate | 768 | +0.3748 | 100 | 0.97 ± 0.02 | 1.00 |
| knn_mean_cosine | null | 2560 | +0.0004 | 100 | 0.99 ± 0.01 | 0.01 |
| knn_mean_cosine | d01 | 2560 | +0.0430 | 100 | 0.96 ± 0.02 | 1.00 |
| knn_mean_cosine | d03 | 2560 | +0.1853 | 100 | 0.99 ± 0.01 | 1.00 |
| knn_mean_cosine | d05 | 2560 | +0.3644 | 100 | 0.95 ± 0.02 | 1.00 |
| knn_mean_cosine | moderate | 2560 | +0.4653 | 100 | 0.96 ± 0.02 | 1.00 |
| mahalanobis_l2 | null | 2560 | +0.0005 | 100 | 0.98 ± 0.01 | 0.03 |
| mahalanobis_l2 | d01 | 2560 | +0.0388 | 100 | 0.99 ± 0.01 | 1.00 |
| mahalanobis_l2 | d03 | 2560 | +0.1750 | 100 | 0.99 ± 0.01 | 1.00 |
| mahalanobis_l2 | d05 | 2560 | +0.3076 | 100 | 0.98 ± 0.01 | 1.00 |
| mahalanobis_l2 | moderate | 2560 | +0.4649 | 100 | 0.96 ± 0.02 | 1.00 |

d768: 10 cells, min n 100; jackknife coverage < 0.93 in 0.

d2560: 10 cells, min n 100; jackknife coverage < 0.93 in 0.

## Real cells beyond the simulated Δ range

Real Δ (item 1, seen = strict) above the largest simulated true Δ of its design and scorer: coverage not verified at this Δ.

| design | scorer | max simulated true Δ | real rows | rows above (coverage not verified at this Δ) |
|---|---|---|---|---|
| breakhis | knn_mean_cosine | +0.1290 | 85 | 42 |
| breakhis | mahalanobis_l2 | +0.1367 | 85 | 60 |
| camelyon | knn_mean_cosine | +0.1416 | 27 | 5 |
| camelyon | mahalanobis_l2 | +0.1445 | 27 | 3 |
| dermamnist | knn_mean_cosine | +0.0887 | 17 | 0 |
| dermamnist | mahalanobis_l2 | +0.0364 | 17 | 0 |
| isic2019 | knn_mean_cosine | +0.1483 | 17 | 0 |
| isic2019 | mahalanobis_l2 | +0.0563 | 17 | 3 |
| kermany | knn_mean_cosine | +0.1483 | 12 | 0 |
| kermany | mahalanobis_l2 | +0.1489 | 12 | 0 |

Coverage not verified at this Δ:

- breakhis_densenet121_s42_std_r0 / knn_mean_cosine: Δ 0.1559
- breakhis_densenet121_s42_std_r1 / knn_mean_cosine: Δ 0.2447
- breakhis_densenet121_s42_std_r2 / knn_mean_cosine: Δ 0.1875
- breakhis_densenet121_s42_std_r4 / knn_mean_cosine: Δ 0.1810
- breakhis_densenet121_s43_std_r0 / knn_mean_cosine: Δ 0.1591
- breakhis_densenet121_s43_std_r1 / knn_mean_cosine: Δ 0.2278
- breakhis_densenet121_s43_std_r2 / knn_mean_cosine: Δ 0.1870
- breakhis_densenet121_s43_std_r4 / knn_mean_cosine: Δ 0.1824
- breakhis_effb3_s42_std_r1 / knn_mean_cosine: Δ 0.1571
- breakhis_effb3_s42_std_r2 / knn_mean_cosine: Δ 0.1332
- breakhis_mobilenet_v3_large_s42_std_r1 / knn_mean_cosine: Δ 0.2028
- breakhis_mobilenet_v3_large_s42_std_r2 / knn_mean_cosine: Δ 0.1565
- breakhis_mobilenet_v3_large_s42_std_r4 / knn_mean_cosine: Δ 0.1551
- breakhis_regnet_y_3_2gf_s42_std_r0 / knn_mean_cosine: Δ 0.1844
- breakhis_regnet_y_3_2gf_s42_std_r1 / knn_mean_cosine: Δ 0.2768
- breakhis_regnet_y_3_2gf_s42_std_r2 / knn_mean_cosine: Δ 0.2157
- breakhis_regnet_y_3_2gf_s42_std_r3 / knn_mean_cosine: Δ 0.1541
- breakhis_regnet_y_3_2gf_s42_std_r4 / knn_mean_cosine: Δ 0.2379
- breakhis_resnet18_s42_std_r0 / knn_mean_cosine: Δ 0.1408
- breakhis_resnet18_s42_std_r1 / knn_mean_cosine: Δ 0.2179
- breakhis_resnet18_s42_std_r2 / knn_mean_cosine: Δ 0.1659
- breakhis_resnet18_s42_std_r4 / knn_mean_cosine: Δ 0.1531
- breakhis_resnet18_s43_std_r0 / knn_mean_cosine: Δ 0.1346
- breakhis_resnet18_s43_std_r1 / knn_mean_cosine: Δ 0.2354
- breakhis_resnet18_s43_std_r2 / knn_mean_cosine: Δ 0.1741
- breakhis_resnet18_s43_std_r4 / knn_mean_cosine: Δ 0.1614
- breakhis_resnet50_s42_std_r0 / knn_mean_cosine: Δ 0.1399
- breakhis_resnet50_s42_std_r1 / knn_mean_cosine: Δ 0.2215
- breakhis_resnet50_s42_std_r2 / knn_mean_cosine: Δ 0.1883
- breakhis_resnet50_s42_std_r3 / knn_mean_cosine: Δ 0.1334
- breakhis_resnet50_s42_std_r4 / knn_mean_cosine: Δ 0.1840
- breakhis_resnet50_s43_std_r0 / knn_mean_cosine: Δ 0.1526
- breakhis_resnet50_s43_std_r1 / knn_mean_cosine: Δ 0.2202
- breakhis_resnet50_s43_std_r2 / knn_mean_cosine: Δ 0.1945
- breakhis_resnet50_s43_std_r4 / knn_mean_cosine: Δ 0.1777
- breakhis_fm_uni_s42_std_r0 / knn_mean_cosine: Δ 0.1508
- breakhis_fm_uni_s42_std_r1 / knn_mean_cosine: Δ 0.2184
- breakhis_fm_uni_s42_std_r2 / knn_mean_cosine: Δ 0.1843
- breakhis_fm_uni_s42_std_r3 / knn_mean_cosine: Δ 0.1506
- breakhis_fm_uni_s42_std_r4 / knn_mean_cosine: Δ 0.1677
- breakhis_fm_virchow2_s42_std_r1 / knn_mean_cosine: Δ 0.1424
- breakhis_fm_virchow2_s42_std_r2 / knn_mean_cosine: Δ 0.1304
- breakhis_convnext_tiny_s42_std_r1 / mahalanobis_l2: Δ 0.1560
- breakhis_convnext_tiny_s43_std_r1 / mahalanobis_l2: Δ 0.1673
- breakhis_densenet121_s42_std_r0 / mahalanobis_l2: Δ 0.2155
- breakhis_densenet121_s42_std_r1 / mahalanobis_l2: Δ 0.3241
- breakhis_densenet121_s42_std_r2 / mahalanobis_l2: Δ 0.2670
- breakhis_densenet121_s42_std_r3 / mahalanobis_l2: Δ 0.2029
- breakhis_densenet121_s42_std_r4 / mahalanobis_l2: Δ 0.2552
- breakhis_densenet121_s43_std_r0 / mahalanobis_l2: Δ 0.2279
- breakhis_densenet121_s43_std_r1 / mahalanobis_l2: Δ 0.3174
- breakhis_densenet121_s43_std_r2 / mahalanobis_l2: Δ 0.2641
- breakhis_densenet121_s43_std_r3 / mahalanobis_l2: Δ 0.1962
- breakhis_densenet121_s43_std_r4 / mahalanobis_l2: Δ 0.2448
- breakhis_effb3_s42_std_r0 / mahalanobis_l2: Δ 0.1430
- breakhis_effb3_s42_std_r1 / mahalanobis_l2: Δ 0.2310
- breakhis_effb3_s42_std_r2 / mahalanobis_l2: Δ 0.1890
- breakhis_effb3_s42_std_r3 / mahalanobis_l2: Δ 0.1412
- breakhis_effb3_s42_std_r4 / mahalanobis_l2: Δ 0.1713
- breakhis_efficientnet_v2_s_s42_std_r1 / mahalanobis_l2: Δ 0.1587
- breakhis_efficientnet_v2_s_s42_std_r2 / mahalanobis_l2: Δ 0.1398
- breakhis_mobilenet_v3_large_s42_std_r0 / mahalanobis_l2: Δ 0.1398
- breakhis_mobilenet_v3_large_s42_std_r1 / mahalanobis_l2: Δ 0.2384
- breakhis_mobilenet_v3_large_s42_std_r2 / mahalanobis_l2: Δ 0.1849
- breakhis_mobilenet_v3_large_s42_std_r3 / mahalanobis_l2: Δ 0.1463
- breakhis_mobilenet_v3_large_s42_std_r4 / mahalanobis_l2: Δ 0.1848
- breakhis_regnet_y_3_2gf_s42_std_r0 / mahalanobis_l2: Δ 0.2461
- breakhis_regnet_y_3_2gf_s42_std_r1 / mahalanobis_l2: Δ 0.3143
- breakhis_regnet_y_3_2gf_s42_std_r2 / mahalanobis_l2: Δ 0.2541
- breakhis_regnet_y_3_2gf_s42_std_r3 / mahalanobis_l2: Δ 0.2142
- breakhis_regnet_y_3_2gf_s42_std_r4 / mahalanobis_l2: Δ 0.2833
- breakhis_resnet18_s42_std_r0 / mahalanobis_l2: Δ 0.1845
- breakhis_resnet18_s42_std_r1 / mahalanobis_l2: Δ 0.2726
- breakhis_resnet18_s42_std_r2 / mahalanobis_l2: Δ 0.2075
- breakhis_resnet18_s42_std_r3 / mahalanobis_l2: Δ 0.1613
- breakhis_resnet18_s42_std_r4 / mahalanobis_l2: Δ 0.2011
- breakhis_resnet18_s43_std_r0 / mahalanobis_l2: Δ 0.1868
- breakhis_resnet18_s43_std_r1 / mahalanobis_l2: Δ 0.2826
- breakhis_resnet18_s43_std_r2 / mahalanobis_l2: Δ 0.2178
- breakhis_resnet18_s43_std_r3 / mahalanobis_l2: Δ 0.1639
- breakhis_resnet18_s43_std_r4 / mahalanobis_l2: Δ 0.1997
- breakhis_resnet50_s42_std_r0 / mahalanobis_l2: Δ 0.2269
- breakhis_resnet50_s42_std_r1 / mahalanobis_l2: Δ 0.3237
- breakhis_resnet50_s42_std_r2 / mahalanobis_l2: Δ 0.2777
- breakhis_resnet50_s42_std_r3 / mahalanobis_l2: Δ 0.2152
- breakhis_resnet50_s42_std_r4 / mahalanobis_l2: Δ 0.2784
- breakhis_resnet50_s43_std_r0 / mahalanobis_l2: Δ 0.2364
- breakhis_resnet50_s43_std_r1 / mahalanobis_l2: Δ 0.3213
- breakhis_resnet50_s43_std_r2 / mahalanobis_l2: Δ 0.2788
- breakhis_resnet50_s43_std_r3 / mahalanobis_l2: Δ 0.2105
- breakhis_resnet50_s43_std_r4 / mahalanobis_l2: Δ 0.2750
- breakhis_fm_conch_v1_5_s42_std_r1 / mahalanobis_l2: Δ 0.1411
- breakhis_fm_uni_s42_std_r0 / mahalanobis_l2: Δ 0.2191
- breakhis_fm_uni_s42_std_r1 / mahalanobis_l2: Δ 0.2863
- breakhis_fm_uni_s42_std_r2 / mahalanobis_l2: Δ 0.2455
- breakhis_fm_uni_s42_std_r3 / mahalanobis_l2: Δ 0.2279
- breakhis_fm_uni_s42_std_r4 / mahalanobis_l2: Δ 0.2438
- breakhis_fm_virchow2_s42_std_r0 / mahalanobis_l2: Δ 0.2026
- breakhis_fm_virchow2_s42_std_r1 / mahalanobis_l2: Δ 0.2501
- breakhis_fm_virchow2_s42_std_r2 / mahalanobis_l2: Δ 0.2309
- breakhis_fm_virchow2_s42_std_r3 / mahalanobis_l2: Δ 0.2024
- breakhis_fm_virchow2_s42_std_r4 / mahalanobis_l2: Δ 0.2128
- camelyon_resnet50_s42 / knn_mean_cosine: Δ 0.2033
- camelyon_virchow2_s43 / knn_mean_cosine: Δ 0.1525
- camelyon_conch_v1_5_s42 / knn_mean_cosine: Δ 0.1505
- camelyon_conch_v1_5_s43 / knn_mean_cosine: Δ 0.1907
- camelyon_conch_v1_5_s44 / knn_mean_cosine: Δ 0.1451
- camelyon_resnet50_s42 / mahalanobis_l2: Δ 0.2156
- camelyon_regnet_y_3_2gf_s42 / mahalanobis_l2: Δ 0.1492
- camelyon_resnet18_s42 / mahalanobis_l2: Δ 0.1683
- isic2019_regnet_y_3_2gf_s42_std / mahalanobis_l2: Δ 0.0694
- isic2019_resnet50_s42_std / mahalanobis_l2: Δ 0.0600
- isic2019_resnet50_s43_std / mahalanobis_l2: Δ 0.0578

## Caveats

1. The local-v3 vs HPC comparison is dropped: the local run used the package scorers, this run the Track A scorers, so the cells are not comparable.
2. Coverage is verified for seen = strict only; item-1 rows with seen = tracka are not covered by this study.
3. ViM is not studied: Track A ViM is not reproduced (item 1 match table).
4. v3 scaling kept: Camelyon sizes 1/30; Kermany train 1/2 and OOD 1/8.
