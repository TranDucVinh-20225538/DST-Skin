# R4 analysis 1: threshold repair by group-disjoint calibration (EXPLORATORY)

Commit at report time: `a2dee75`. Precommit: `results/r4/1/PRECOMMIT.json` (committed `3e369df`, before any run).

**Verdict: prediction PARTLY SUPPORTED** (both parts hold in 4 of 14 cells; disjoint median coverage in [0.93, 0.97]: 8 of 14; leaky median coverage < 0.93: 10 of 14).

Camelyon17, seed 42, 30 training slides, 20 hospital-stratified slide partitions F / C / T (10 / 10 / 10 slides, 4 + 3 + 3 per hospital).
The scorer is refit on F's training patches (cached frozen features; the cached Track C scores come from 2-fold fits and cannot be reused).
Leaky threshold = 5th percentile of scores on F's id_val patches. Disjoint threshold = 5th percentile on C's id_val patches.
Both are applied to T's id_val patches. Values are the median [min, max] over the 20 partitions.

## (a) Patch-level realised coverage on T (target 0.95)

| backbone | scorer | n_groups_fit | K | d | ID acc | leaky | disjoint | D | L |
|---|---|---|---|---|---|---|---|---|---|
| uni | mahalanobis_l2 | 10 | 3-way x 20 | 1024 | 0.9944 | 0.714 [0.363, 0.914] | 0.973 [0.537, 0.998] | no | yes |
| uni | knn_mean_cosine | 10 | 3-way x 20 | 1024 | 0.9944 | 0.577 [0.275, 0.902] | 0.989 [0.464, 1.000] | no | yes |
| virchow2 | mahalanobis_l2 | 10 | 3-way x 20 | 2560 | 0.9948 | 0.832 [0.530, 0.940] | 0.960 [0.842, 0.991] | yes | yes |
| virchow2 | knn_mean_cosine | 10 | 3-way x 20 | 2560 | 0.9948 | 0.798 [0.447, 0.935] | 0.976 [0.627, 0.992] | no | yes |
| dinov2_vitb14 | mahalanobis_l2 | 10 | 3-way x 20 | 768 | 0.9717 | 0.948 [0.899, 0.979] | 0.954 [0.870, 0.977] | yes | no |
| dinov2_vitb14 | knn_mean_cosine | 10 | 3-way x 20 | 768 | 0.9717 | 0.948 [0.897, 0.979] | 0.951 [0.869, 0.978] | yes | no |
| dinov2_vitl14 | mahalanobis_l2 | 10 | 3-way x 20 | 1024 | 0.9765 | 0.950 [0.886, 0.977] | 0.953 [0.861, 0.978] | yes | no |
| dinov2_vitl14 | knn_mean_cosine | 10 | 3-way x 20 | 1024 | 0.9765 | 0.951 [0.885, 0.976] | 0.956 [0.856, 0.979] | yes | no |
| conch_v1_5 | mahalanobis_l2 | 10 | 3-way x 20 | 768 | 0.9873 | 0.928 [0.876, 0.958] | 0.956 [0.906, 0.977] | yes | yes |
| conch_v1_5 | knn_mean_cosine | 10 | 3-way x 20 | 768 | 0.9873 | 0.780 [0.450, 0.934] | 0.982 [0.586, 0.997] | no | yes |
| resnet50 | mahalanobis_l2 | 10 | 3-way x 20 | 2048 | 0.9954 | 0.566 [0.257, 0.966] | 0.880 [0.524, 1.000] | no | yes |
| resnet50 | knn_mean_cosine | 10 | 3-way x 20 | 2048 | 0.9954 | 0.702 [0.334, 0.959] | 0.887 [0.539, 1.000] | no | yes |
| convnext_tiny | mahalanobis_l2 | 10 | 3-way x 20 | 768 | 0.9969 | 0.899 [0.766, 0.937] | 0.949 [0.842, 0.997] | yes | yes |
| convnext_tiny | knn_mean_cosine | 10 | 3-way x 20 | 768 | 0.9969 | 0.876 [0.852, 0.927] | 0.948 [0.895, 0.990] | yes | yes |

D: 0.93 <= disjoint median <= 0.97. L: leaky median < 0.93. Coverage on the calibration slides themselves is 0.95 by construction (median 0.9500 to 0.9500 across cells).

## (b) Slide-level false alarms on T (10 T slides per partition)

Per partition: median slide false-alarm rate, fraction of T slides with rate > 10% and > 20%, and the worst slide's rate.
Each entry is the median [min, max] over 20 partitions. Pooled = 5/50/95th percentiles of all 200 slide rates.

| backbone | scorer | threshold | median slide FA | frac > 10% | frac > 20% | worst slide | pooled q05 / q50 / q95 |
|---|---|---|---|---|---|---|---|
| uni | mahalanobis_l2 | leaky | 0.132 [0.061, 0.335] | 0.60 [0.40, 1.00] | 0.30 [0.10, 0.80] | 0.878 [0.280, 0.969] | 0.018 / 0.140 / 0.870 |
| uni | mahalanobis_l2 | disjoint | 0.008 [0.000, 0.043] | 0.10 [0.00, 0.30] | 0.00 [0.00, 0.20] | 0.148 [0.008, 0.791] | 0.000 / 0.009 / 0.285 |
| uni | knn_mean_cosine | leaky | 0.187 [0.098, 0.497] | 0.70 [0.50, 1.00] | 0.45 [0.20, 0.80] | 0.932 [0.310, 0.985] | 0.027 / 0.191 / 0.945 |
| uni | knn_mean_cosine | disjoint | 0.000 [0.000, 0.052] | 0.00 [0.00, 0.40] | 0.00 [0.00, 0.30] | 0.052 [0.000, 0.900] | 0.000 / 0.000 / 0.217 |
| virchow2 | mahalanobis_l2 | leaky | 0.086 [0.069, 0.160] | 0.40 [0.30, 0.70] | 0.20 [0.00, 0.30] | 0.449 [0.174, 0.806] | 0.024 / 0.092 / 0.570 |
| virchow2 | mahalanobis_l2 | disjoint | 0.022 [0.008, 0.069] | 0.10 [0.00, 0.30] | 0.00 [0.00, 0.20] | 0.164 [0.049, 0.327] | 0.003 / 0.020 / 0.183 |
| virchow2 | knn_mean_cosine | leaky | 0.097 [0.068, 0.172] | 0.50 [0.30, 0.80] | 0.20 [0.10, 0.40] | 0.696 [0.216, 0.962] | 0.034 / 0.102 / 0.699 |
| virchow2 | knn_mean_cosine | disjoint | 0.014 [0.000, 0.053] | 0.10 [0.00, 0.30] | 0.00 [0.00, 0.20] | 0.153 [0.024, 0.610] | 0.000 / 0.015 / 0.210 |
| dinov2_vitb14 | mahalanobis_l2 | leaky | 0.062 [0.034, 0.109] | 0.20 [0.00, 0.60] | 0.00 [0.00, 0.10] | 0.180 [0.095, 0.296] | 0.018 / 0.060 / 0.192 |
| dinov2_vitb14 | mahalanobis_l2 | disjoint | 0.059 [0.023, 0.158] | 0.20 [0.00, 0.80] | 0.05 [0.00, 0.30] | 0.203 [0.076, 0.307] | 0.017 / 0.067 / 0.229 |
| dinov2_vitb14 | knn_mean_cosine | leaky | 0.061 [0.031, 0.103] | 0.20 [0.00, 0.50] | 0.00 [0.00, 0.10] | 0.177 [0.095, 0.307] | 0.018 / 0.056 / 0.187 |
| dinov2_vitb14 | knn_mean_cosine | disjoint | 0.056 [0.025, 0.157] | 0.20 [0.00, 0.70] | 0.00 [0.00, 0.30] | 0.177 [0.073, 0.303] | 0.016 / 0.063 / 0.211 |
| dinov2_vitl14 | mahalanobis_l2 | leaky | 0.055 [0.026, 0.105] | 0.20 [0.00, 0.50] | 0.00 [0.00, 0.20] | 0.145 [0.075, 0.285] | 0.014 / 0.055 / 0.186 |
| dinov2_vitl14 | mahalanobis_l2 | disjoint | 0.057 [0.020, 0.171] | 0.20 [0.00, 0.80] | 0.05 [0.00, 0.30] | 0.196 [0.089, 0.283] | 0.012 / 0.062 / 0.208 |
| dinov2_vitl14 | knn_mean_cosine | leaky | 0.060 [0.025, 0.105] | 0.20 [0.00, 0.60] | 0.00 [0.00, 0.20] | 0.158 [0.067, 0.293] | 0.013 / 0.057 / 0.199 |
| dinov2_vitl14 | knn_mean_cosine | disjoint | 0.058 [0.021, 0.178] | 0.25 [0.00, 0.80] | 0.05 [0.00, 0.30] | 0.192 [0.088, 0.313] | 0.012 / 0.063 / 0.205 |
| conch_v1_5 | mahalanobis_l2 | leaky | 0.066 [0.048, 0.105] | 0.20 [0.00, 0.60] | 0.00 [0.00, 0.20] | 0.172 [0.098, 0.345] | 0.021 / 0.070 / 0.185 |
| conch_v1_5 | mahalanobis_l2 | disjoint | 0.053 [0.020, 0.090] | 0.10 [0.00, 0.50] | 0.00 [0.00, 0.00] | 0.132 [0.088, 0.187] | 0.013 / 0.051 / 0.136 |
| conch_v1_5 | knn_mean_cosine | leaky | 0.103 [0.055, 0.194] | 0.50 [0.10, 0.90] | 0.20 [0.00, 0.40] | 0.574 [0.178, 0.938] | 0.027 / 0.100 / 0.588 |
| conch_v1_5 | knn_mean_cosine | disjoint | 0.010 [0.000, 0.054] | 0.00 [0.00, 0.30] | 0.00 [0.00, 0.20] | 0.051 [0.014, 0.720] | 0.000 / 0.012 / 0.211 |
| resnet50 | mahalanobis_l2 | leaky | 0.048 [0.022, 0.291] | 0.30 [0.10, 0.70] | 0.30 [0.00, 0.50] | 0.801 [0.124, 0.955] | 0.009 / 0.044 / 0.808 |
| resnet50 | mahalanobis_l2 | disjoint | 0.000 [0.000, 0.076] | 0.10 [0.00, 0.40] | 0.10 [0.00, 0.30] | 0.236 [0.000, 0.886] | 0.000 / 0.000 / 0.340 |
| resnet50 | knn_mean_cosine | leaky | 0.042 [0.029, 0.131] | 0.25 [0.10, 0.50] | 0.20 [0.00, 0.40] | 0.549 [0.132, 0.925] | 0.013 / 0.042 / 0.576 |
| resnet50 | knn_mean_cosine | disjoint | 0.005 [0.000, 0.085] | 0.10 [0.00, 0.40] | 0.10 [0.00, 0.20] | 0.218 [0.000, 0.862] | 0.000 / 0.006 / 0.242 |
| convnext_tiny | mahalanobis_l2 | leaky | 0.087 [0.061, 0.115] | 0.40 [0.00, 0.60] | 0.05 [0.00, 0.20] | 0.210 [0.084, 0.702] | 0.030 / 0.083 / 0.224 |
| convnext_tiny | mahalanobis_l2 | disjoint | 0.030 [0.002, 0.077] | 0.00 [0.00, 0.30] | 0.00 [0.00, 0.10] | 0.079 [0.010, 0.307] | 0.001 / 0.023 / 0.114 |
| convnext_tiny | knn_mean_cosine | leaky | 0.106 [0.060, 0.150] | 0.50 [0.20, 0.70] | 0.10 [0.00, 0.30] | 0.211 [0.114, 0.591] | 0.035 / 0.106 / 0.231 |
| convnext_tiny | knn_mean_cosine | disjoint | 0.030 [0.007, 0.071] | 0.00 [0.00, 0.40] | 0.00 [0.00, 0.00] | 0.091 [0.030, 0.165] | 0.004 / 0.028 / 0.104 |

Per-partition values (slide lists, thresholds, per-slide rates): `results/r4/1/raw/*.json`; table: `results/r4/1/summary.csv`.

Caveats: exploratory, one seed, one dataset; the 20 partitions share slides, so the ranges are not independent replicates.
