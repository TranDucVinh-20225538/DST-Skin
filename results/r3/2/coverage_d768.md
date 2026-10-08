# paper_2fold coverage study (realistic designs)

Nominal 95%. Coverage ± binomial MC SE. width = mean CI width; SE/SD = mean reported SE / SD of Delta_hat over the truth replicates; lo>0.02 / lo>0 = fraction of datasets whose CI lower bound exceeds 0.02 / 0 (power when true Delta > threshold, false pass otherwise).

## Calibrated ICC -> achieved true Delta (mean of truth replicates ± MC SE)

| design | scorer | level | ICC | true Δ | ± MC SE | SD(Δ̂) | n truth |
|---|---|---|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | 0 | +0.0009 | 0.0001 | 0.0019 | 400 |
| camelyon | knn_mean_cosine | d01 | 0.0097 | +0.0207 | 0.0002 | 0.0032 | 400 |
| camelyon | knn_mean_cosine | d03 | 0.0274 | +0.0770 | 0.0003 | 0.0069 | 400 |
| camelyon | knn_mean_cosine | d05 | 0.0479 | +0.1692 | 0.0005 | 0.0099 | 400 |
| camelyon | knn_mean_cosine | moderate | 0.1097 | +0.4017 | 0.0010 | 0.0195 | 400 |
| camelyon | mahalanobis_l2 | null | 0 | +0.0009 | 0.0001 | 0.0012 | 400 |
| camelyon | mahalanobis_l2 | d01 | 0.0099 | +0.0191 | 0.0001 | 0.0026 | 400 |
| camelyon | mahalanobis_l2 | d03 | 0.0261 | +0.0758 | 0.0003 | 0.0055 | 400 |
| camelyon | mahalanobis_l2 | d05 | 0.0425 | +0.1544 | 0.0004 | 0.0075 | 400 |
| camelyon | mahalanobis_l2 | moderate | 0.1085 | +0.3748 | 0.0007 | 0.0150 | 400 |

## Coverage per cell

| design | scorer | level | true Δ | method | n | coverage ± MC SE | width | SE/SD | lo>0.02 | lo>0 | s/call |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | +0.0009 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0106 | 1.40 | 0.00 | 0.02 | 88.29 |
| camelyon | knn_mean_cosine | null | +0.0009 | bootstrap old (OOD images) | 100 | 0.85 ± 0.04 | 0.0064 | 0.85 | 0.00 | 0.14 | 3.47 |
| camelyon | knn_mean_cosine | null | +0.0009 | bootstrap (OOD groups) | 100 | 0.85 ± 0.04 | 0.0064 | 0.85 | 0.00 | 0.14 | 3.45 |
| camelyon | knn_mean_cosine | d01 | +0.0207 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0237 | 1.90 | 0.00 | 1.00 | 90.18 |
| camelyon | knn_mean_cosine | d01 | +0.0207 | bootstrap old (OOD images) | 100 | 0.91 ± 0.03 | 0.0208 | 1.74 | 0.00 | 1.00 | 3.48 |
| camelyon | knn_mean_cosine | d01 | +0.0207 | bootstrap (OOD groups) | 100 | 0.91 ± 0.03 | 0.0208 | 1.74 | 0.00 | 1.00 | 3.52 |
| camelyon | knn_mean_cosine | d03 | +0.0770 | jackknife (default) | 100 | 0.93 ± 0.03 | 0.0588 | 2.16 | 1.00 | 1.00 | 93.30 |
| camelyon | knn_mean_cosine | d03 | +0.0770 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0625 | 2.40 | 1.00 | 1.00 | 3.60 |
| camelyon | knn_mean_cosine | d03 | +0.0770 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0625 | 2.40 | 1.00 | 1.00 | 3.68 |
| camelyon | knn_mean_cosine | d05 | +0.1692 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0845 | 2.18 | 1.00 | 1.00 | 93.99 |
| camelyon | knn_mean_cosine | d05 | +0.1692 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.1013 | 2.70 | 1.00 | 1.00 | 3.63 |
| camelyon | knn_mean_cosine | d05 | +0.1692 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.1013 | 2.70 | 1.00 | 1.00 | 3.61 |
| camelyon | knn_mean_cosine | moderate | +0.4017 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0861 | 1.13 | 1.00 | 1.00 | 89.38 |
| camelyon | knn_mean_cosine | moderate | +0.4017 | bootstrap old (OOD images) | 100 | 0.90 ± 0.03 | 0.0644 | 0.86 | 1.00 | 1.00 | 3.37 |
| camelyon | knn_mean_cosine | moderate | +0.4017 | bootstrap (OOD groups) | 100 | 0.90 ± 0.03 | 0.0644 | 0.86 | 1.00 | 1.00 | 3.43 |
| camelyon | mahalanobis_l2 | null | +0.0009 | jackknife (default) | 100 | 0.93 ± 0.03 | 0.0070 | 1.49 | 0.00 | 0.01 | 52.70 |
| camelyon | mahalanobis_l2 | null | +0.0009 | bootstrap old (OOD images) | 100 | 0.75 ± 0.04 | 0.0032 | 0.69 | 0.00 | 0.07 | 2.10 |
| camelyon | mahalanobis_l2 | null | +0.0009 | bootstrap (OOD groups) | 100 | 0.75 ± 0.04 | 0.0032 | 0.69 | 0.00 | 0.07 | 2.19 |
| camelyon | mahalanobis_l2 | d01 | +0.0191 | jackknife (default) | 100 | 0.95 ± 0.02 | 0.0217 | 2.09 | 0.00 | 1.00 | 51.07 |
| camelyon | mahalanobis_l2 | d01 | +0.0191 | bootstrap old (OOD images) | 100 | 0.94 ± 0.02 | 0.0182 | 1.84 | 0.00 | 1.00 | 2.10 |
| camelyon | mahalanobis_l2 | d01 | +0.0191 | bootstrap (OOD groups) | 100 | 0.94 ± 0.02 | 0.0182 | 1.84 | 0.00 | 1.00 | 2.09 |
| camelyon | mahalanobis_l2 | d03 | +0.0758 | jackknife (default) | 100 | 0.97 ± 0.02 | 0.0578 | 2.66 | 1.00 | 1.00 | 51.35 |
| camelyon | mahalanobis_l2 | d03 | +0.0758 | bootstrap old (OOD images) | 100 | 0.94 ± 0.02 | 0.0448 | 2.14 | 1.00 | 1.00 | 2.11 |
| camelyon | mahalanobis_l2 | d03 | +0.0758 | bootstrap (OOD groups) | 100 | 0.94 ± 0.02 | 0.0448 | 2.14 | 1.00 | 1.00 | 2.08 |
| camelyon | mahalanobis_l2 | d05 | +0.1544 | jackknife (default) | 100 | 0.97 ± 0.02 | 0.0645 | 2.19 | 1.00 | 1.00 | 50.83 |
| camelyon | mahalanobis_l2 | d05 | +0.1544 | bootstrap old (OOD images) | 100 | 0.95 ± 0.02 | 0.0548 | 1.92 | 1.00 | 1.00 | 2.09 |
| camelyon | mahalanobis_l2 | d05 | +0.1544 | bootstrap (OOD groups) | 100 | 0.95 ± 0.02 | 0.0548 | 1.92 | 1.00 | 1.00 | 2.07 |
| camelyon | mahalanobis_l2 | moderate | +0.3748 | jackknife (default) | 100 | 0.97 ± 0.02 | 0.0696 | 1.18 | 1.00 | 1.00 | 51.56 |
| camelyon | mahalanobis_l2 | moderate | +0.3748 | bootstrap old (OOD images) | 100 | 0.93 ± 0.03 | 0.0563 | 0.98 | 1.00 | 1.00 | 2.11 |
| camelyon | mahalanobis_l2 | moderate | +0.3748 | bootstrap (OOD groups) | 100 | 0.93 ± 0.03 | 0.0563 | 0.98 | 1.00 | 1.00 | 2.09 |

## Compact: coverage by method (rows = cells)

Cells: n datasets (jackknife) in brackets; MC SE of 0.95 is 0.022 at n=100, 0.031 at n=50.

| design | scorer | level | true Δ | jackknife [n] | bootstrap old | bootstrap OOD-groups | subsample-normal [n] |
|---|---|---|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | +0.0009 | 0.98 ± 0.01 [100] | 0.85 ± 0.04 | 0.85 ± 0.04 | — |
| camelyon | knn_mean_cosine | d01 | +0.0207 | 0.96 ± 0.02 [100] | 0.91 ± 0.03 | 0.91 ± 0.03 | — |
| camelyon | knn_mean_cosine | d03 | +0.0770 | 0.93 ± 0.03 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| camelyon | knn_mean_cosine | d05 | +0.1692 | 0.96 ± 0.02 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| camelyon | knn_mean_cosine | moderate | +0.4017 | 0.98 ± 0.01 [100] | 0.90 ± 0.03 | 0.90 ± 0.03 | — |
| camelyon | mahalanobis_l2 | null | +0.0009 | 0.93 ± 0.03 [100] | 0.75 ± 0.04 | 0.75 ± 0.04 | — |
| camelyon | mahalanobis_l2 | d01 | +0.0191 | 0.95 ± 0.02 [100] | 0.94 ± 0.02 | 0.94 ± 0.02 | — |
| camelyon | mahalanobis_l2 | d03 | +0.0758 | 0.97 ± 0.02 [100] | 0.94 ± 0.02 | 0.94 ± 0.02 | — |
| camelyon | mahalanobis_l2 | d05 | +0.1544 | 0.97 ± 0.02 [100] | 0.95 ± 0.02 | 0.95 ± 0.02 | — |
| camelyon | mahalanobis_l2 | moderate | +0.3748 | 0.97 ± 0.02 [100] | 0.93 ± 0.03 | 0.93 ± 0.03 | — |

## Range per scorer x method (all designs and levels)

| scorer | method | coverage min–max | SE/SD min–max | cells |
|---|---|---|---|---:|

## Pre-registered decision (jackknife, threshold 0.93, fixed before results)

Cells: 10; jackknife coverage min = 0.93; cells < 0.93: 0.

**Outcome: jackknife coverage >= 0.93 in all cells -> recommend jackknife CIs for the paper.**
