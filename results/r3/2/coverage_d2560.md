# paper_2fold coverage study (realistic designs)

Nominal 95%. Coverage ± binomial MC SE. width = mean CI width; SE/SD = mean reported SE / SD of Delta_hat over the truth replicates; lo>0.02 / lo>0 = fraction of datasets whose CI lower bound exceeds 0.02 / 0 (power when true Delta > threshold, false pass otherwise).

## Calibrated ICC -> achieved true Delta (mean of truth replicates ± MC SE)

| design | scorer | level | ICC | true Δ | ± MC SE | SD(Δ̂) | n truth |
|---|---|---|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | 0 | +0.0004 | 0.0001 | 0.0016 | 400 |
| camelyon | knn_mean_cosine | d01 | 0.0097 | +0.0430 | 0.0002 | 0.0046 | 400 |
| camelyon | knn_mean_cosine | d03 | 0.0274 | +0.1853 | 0.0004 | 0.0087 | 400 |
| camelyon | knn_mean_cosine | d05 | 0.0479 | +0.3644 | 0.0006 | 0.0123 | 400 |
| camelyon | knn_mean_cosine | moderate | 0.1097 | +0.4653 | 0.0010 | 0.0194 | 400 |
| camelyon | mahalanobis_l2 | null | 0 | +0.0005 | 0.0000 | 0.0007 | 400 |
| camelyon | mahalanobis_l2 | d01 | 0.0099 | +0.0388 | 0.0002 | 0.0035 | 400 |
| camelyon | mahalanobis_l2 | d03 | 0.0261 | +0.1750 | 0.0004 | 0.0072 | 400 |
| camelyon | mahalanobis_l2 | d05 | 0.0425 | +0.3076 | 0.0004 | 0.0084 | 400 |
| camelyon | mahalanobis_l2 | moderate | 0.1085 | +0.4649 | 0.0008 | 0.0160 | 400 |

## Coverage per cell

| design | scorer | level | true Δ | method | n | coverage ± MC SE | width | SE/SD | lo>0.02 | lo>0 | s/call |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | +0.0004 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0092 | 1.42 | 0.00 | 0.01 | 219.68 |
| camelyon | knn_mean_cosine | null | +0.0004 | bootstrap old (OOD images) | 100 | 0.93 ± 0.03 | 0.0061 | 0.94 | 0.00 | 0.02 | 7.93 |
| camelyon | knn_mean_cosine | null | +0.0004 | bootstrap (OOD groups) | 100 | 0.93 ± 0.03 | 0.0061 | 0.94 | 0.00 | 0.02 | 7.96 |
| camelyon | knn_mean_cosine | d01 | +0.0430 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0384 | 2.12 | 0.61 | 1.00 | 236.44 |
| camelyon | knn_mean_cosine | d01 | +0.0430 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0388 | 2.25 | 0.21 | 1.00 | 8.32 |
| camelyon | knn_mean_cosine | d01 | +0.0430 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0388 | 2.25 | 0.21 | 1.00 | 8.29 |
| camelyon | knn_mean_cosine | d03 | +0.1853 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0876 | 2.58 | 1.00 | 1.00 | 230.71 |
| camelyon | knn_mean_cosine | d03 | +0.1853 | bootstrap old (OOD images) | 100 | 1.00 ± 0.00 | 0.1098 | 3.35 | 1.00 | 1.00 | 8.31 |
| camelyon | knn_mean_cosine | d03 | +0.1853 | bootstrap (OOD groups) | 100 | 1.00 ± 0.00 | 0.1098 | 3.35 | 1.00 | 1.00 | 8.29 |
| camelyon | knn_mean_cosine | d05 | +0.3644 | jackknife (default) | 100 | 0.95 ± 0.02 | 0.0648 | 1.34 | 1.00 | 1.00 | 229.90 |
| camelyon | knn_mean_cosine | d05 | +0.3644 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0858 | 1.82 | 1.00 | 1.00 | 8.16 |
| camelyon | knn_mean_cosine | d05 | +0.3644 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0858 | 1.82 | 1.00 | 1.00 | 8.18 |
| camelyon | knn_mean_cosine | moderate | +0.4653 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0803 | 1.06 | 1.00 | 1.00 | 223.36 |
| camelyon | knn_mean_cosine | moderate | +0.4653 | bootstrap old (OOD images) | 100 | 0.92 ± 0.03 | 0.0649 | 0.86 | 1.00 | 1.00 | 7.70 |
| camelyon | knn_mean_cosine | moderate | +0.4653 | bootstrap (OOD groups) | 100 | 0.92 ± 0.03 | 0.0649 | 0.86 | 1.00 | 1.00 | 7.70 |
| camelyon | mahalanobis_l2 | null | +0.0005 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0044 | 1.57 | 0.00 | 0.03 | 1076.52 |
| camelyon | mahalanobis_l2 | null | +0.0005 | bootstrap old (OOD images) | 100 | 0.89 ± 0.03 | 0.0021 | 0.75 | 0.00 | 0.11 | 35.59 |
| camelyon | mahalanobis_l2 | null | +0.0005 | bootstrap (OOD groups) | 100 | 0.89 ± 0.03 | 0.0021 | 0.75 | 0.00 | 0.11 | 35.70 |
| camelyon | mahalanobis_l2 | d01 | +0.0388 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0377 | 2.72 | 0.47 | 1.00 | 1118.92 |
| camelyon | mahalanobis_l2 | d01 | +0.0388 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0321 | 2.44 | 0.11 | 1.00 | 36.69 |
| camelyon | mahalanobis_l2 | d01 | +0.0388 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0321 | 2.44 | 0.11 | 1.00 | 36.71 |
| camelyon | mahalanobis_l2 | d03 | +0.1750 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0772 | 2.74 | 1.00 | 1.00 | 1101.80 |
| camelyon | mahalanobis_l2 | d03 | +0.1750 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0586 | 2.13 | 1.00 | 1.00 | 36.49 |
| camelyon | mahalanobis_l2 | d03 | +0.1750 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0586 | 2.13 | 1.00 | 1.00 | 36.69 |
| camelyon | mahalanobis_l2 | d05 | +0.3076 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0532 | 1.62 | 1.00 | 1.00 | 1141.93 |
| camelyon | mahalanobis_l2 | d05 | +0.3076 | bootstrap old (OOD images) | 100 | 0.93 ± 0.03 | 0.0485 | 1.50 | 1.00 | 1.00 | 37.38 |
| camelyon | mahalanobis_l2 | d05 | +0.3076 | bootstrap (OOD groups) | 100 | 0.93 ± 0.03 | 0.0485 | 1.50 | 1.00 | 1.00 | 37.37 |
| camelyon | mahalanobis_l2 | moderate | +0.4649 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0766 | 1.22 | 1.00 | 1.00 | 1189.69 |
| camelyon | mahalanobis_l2 | moderate | +0.4649 | bootstrap old (OOD images) | 100 | 0.92 ± 0.03 | 0.0579 | 0.93 | 1.00 | 1.00 | 39.79 |
| camelyon | mahalanobis_l2 | moderate | +0.4649 | bootstrap (OOD groups) | 100 | 0.92 ± 0.03 | 0.0579 | 0.93 | 1.00 | 1.00 | 39.46 |

## Compact: coverage by method (rows = cells)

Cells: n datasets (jackknife) in brackets; MC SE of 0.95 is 0.022 at n=100, 0.031 at n=50.

| design | scorer | level | true Δ | jackknife [n] | bootstrap old | bootstrap OOD-groups | subsample-normal [n] |
|---|---|---|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | +0.0004 | 0.99 ± 0.01 [100] | 0.93 ± 0.03 | 0.93 ± 0.03 | — |
| camelyon | knn_mean_cosine | d01 | +0.0430 | 0.96 ± 0.02 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| camelyon | knn_mean_cosine | d03 | +0.1853 | 0.99 ± 0.01 [100] | 1.00 ± 0.00 | 1.00 ± 0.00 | — |
| camelyon | knn_mean_cosine | d05 | +0.3644 | 0.95 ± 0.02 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| camelyon | knn_mean_cosine | moderate | +0.4653 | 0.96 ± 0.02 [100] | 0.92 ± 0.03 | 0.92 ± 0.03 | — |
| camelyon | mahalanobis_l2 | null | +0.0005 | 0.98 ± 0.01 [100] | 0.89 ± 0.03 | 0.89 ± 0.03 | — |
| camelyon | mahalanobis_l2 | d01 | +0.0388 | 0.99 ± 0.01 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| camelyon | mahalanobis_l2 | d03 | +0.1750 | 0.99 ± 0.01 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| camelyon | mahalanobis_l2 | d05 | +0.3076 | 0.98 ± 0.01 [100] | 0.93 ± 0.03 | 0.93 ± 0.03 | — |
| camelyon | mahalanobis_l2 | moderate | +0.4649 | 0.96 ± 0.02 [100] | 0.92 ± 0.03 | 0.92 ± 0.03 | — |

## Range per scorer x method (all designs and levels)

| scorer | method | coverage min–max | SE/SD min–max | cells |
|---|---|---|---|---:|

## Pre-registered decision (jackknife, threshold 0.93, fixed before results)

Cells: 10; jackknife coverage min = 0.95; cells < 0.93: 0.

**Outcome: jackknife coverage >= 0.93 in all cells -> recommend jackknife CIs for the paper.**
