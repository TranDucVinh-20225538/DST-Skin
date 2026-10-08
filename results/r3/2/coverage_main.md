# paper_2fold coverage study (realistic designs)

Nominal 95%. Coverage ± binomial MC SE. width = mean CI width; SE/SD = mean reported SE / SD of Delta_hat over the truth replicates; lo>0.02 / lo>0 = fraction of datasets whose CI lower bound exceeds 0.02 / 0 (power when true Delta > threshold, false pass otherwise).

## Calibrated ICC -> achieved true Delta (mean of truth replicates ± MC SE)

| design | scorer | level | ICC | true Δ | ± MC SE | SD(Δ̂) | n truth |
|---|---|---|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | 0 | +0.0024 | 0.0002 | 0.0032 | 400 |
| camelyon | knn_mean_cosine | d01 | 0.0097 | +0.0093 | 0.0002 | 0.0036 | 400 |
| camelyon | knn_mean_cosine | d03 | 0.0274 | +0.0249 | 0.0002 | 0.0049 | 400 |
| camelyon | knn_mean_cosine | d05 | 0.0479 | +0.0476 | 0.0003 | 0.0069 | 400 |
| camelyon | knn_mean_cosine | moderate | 0.1097 | +0.1416 | 0.0007 | 0.0139 | 400 |
| camelyon | mahalanobis_l2 | null | 0 | +0.0022 | 0.0001 | 0.0028 | 400 |
| camelyon | mahalanobis_l2 | d01 | 0.0099 | +0.0093 | 0.0002 | 0.0030 | 400 |
| camelyon | mahalanobis_l2 | d03 | 0.0261 | +0.0245 | 0.0002 | 0.0040 | 400 |
| camelyon | mahalanobis_l2 | d05 | 0.0425 | +0.0448 | 0.0003 | 0.0054 | 400 |
| camelyon | mahalanobis_l2 | moderate | 0.1085 | +0.1445 | 0.0006 | 0.0111 | 400 |
| breakhis | knn_mean_cosine | null | 0 | +0.0006 | 0.0001 | 0.0019 | 400 |
| breakhis | knn_mean_cosine | d01 | 0.0293 | +0.0080 | 0.0001 | 0.0022 | 400 |
| breakhis | knn_mean_cosine | d03 | 0.0657 | +0.0243 | 0.0001 | 0.0030 | 400 |
| breakhis | knn_mean_cosine | d05 | 0.0982 | +0.0485 | 0.0002 | 0.0041 | 400 |
| breakhis | knn_mean_cosine | moderate | 0.1629 | +0.1290 | 0.0004 | 0.0080 | 400 |
| breakhis | mahalanobis_l2 | null | 0 | +0.0007 | 0.0001 | 0.0011 | 400 |
| breakhis | mahalanobis_l2 | d01 | 0.0306 | +0.0079 | 0.0001 | 0.0016 | 400 |
| breakhis | mahalanobis_l2 | d03 | 0.0629 | +0.0238 | 0.0001 | 0.0027 | 400 |
| breakhis | mahalanobis_l2 | d05 | 0.0874 | +0.0444 | 0.0002 | 0.0038 | 400 |
| breakhis | mahalanobis_l2 | moderate | 0.161 | +0.1367 | 0.0004 | 0.0082 | 400 |
| dermamnist | knn_mean_cosine | null | 0 | +0.0001 | 0.0001 | 0.0018 | 400 |
| dermamnist | knn_mean_cosine | d01 | 0.2611 | +0.0079 | 0.0001 | 0.0018 | 400 |
| dermamnist | knn_mean_cosine | d03 | 0.451 | +0.0297 | 0.0001 | 0.0019 | 400 |
| dermamnist | knn_mean_cosine | d05 | 0.6123 | +0.0496 | 0.0001 | 0.0021 | 400 |
| dermamnist | knn_mean_cosine | max | 0.95 | +0.0887 | 0.0002 | 0.0032 | 400 |
| dermamnist | mahalanobis_l2 | null | 0 | +0.0001 | 0.0000 | 0.0007 | 400 |
| dermamnist | mahalanobis_l2 | d01 | 0.5571 | +0.0093 | 0.0001 | 0.0010 | 400 |
| dermamnist | mahalanobis_l2 | d03 | 0.8809 | +0.0298 | 0.0001 | 0.0018 | 400 |
| dermamnist | mahalanobis_l2 | max | 0.95 | +0.0364 | 0.0001 | 0.0020 | 400 |
| isic2019 | knn_mean_cosine | null | 0 | +0.0000 | 0.0000 | 0.0008 | 400 |
| isic2019 | knn_mean_cosine | d01 | 0.2204 | +0.0072 | 0.0000 | 0.0009 | 400 |
| isic2019 | knn_mean_cosine | d03 | 0.3323 | +0.0276 | 0.0001 | 0.0012 | 400 |
| isic2019 | knn_mean_cosine | d05 | 0.4073 | +0.0477 | 0.0001 | 0.0016 | 400 |
| isic2019 | knn_mean_cosine | moderate | 0.8676 | +0.1483 | 0.0002 | 0.0047 | 400 |
| isic2019 | mahalanobis_l2 | null | 0 | +0.0000 | 0.0000 | 0.0004 | 400 |
| isic2019 | mahalanobis_l2 | d01 | 0.4356 | +0.0092 | 0.0000 | 0.0007 | 400 |
| isic2019 | mahalanobis_l2 | d03 | 0.7184 | +0.0294 | 0.0001 | 0.0013 | 400 |
| isic2019 | mahalanobis_l2 | d05 | 0.8969 | +0.0493 | 0.0001 | 0.0019 | 400 |
| isic2019 | mahalanobis_l2 | max | 0.95 | +0.0563 | 0.0001 | 0.0022 | 400 |
| kermany | knn_mean_cosine | null | 0 | -0.0000 | 0.0001 | 0.0015 | 400 |
| kermany | knn_mean_cosine | d01 | 0.1105 | +0.0066 | 0.0001 | 0.0017 | 400 |
| kermany | knn_mean_cosine | d03 | 0.1673 | +0.0219 | 0.0001 | 0.0023 | 400 |
| kermany | knn_mean_cosine | d05 | 0.2106 | +0.0450 | 0.0002 | 0.0033 | 400 |
| kermany | knn_mean_cosine | moderate | 0.332 | +0.1483 | 0.0004 | 0.0079 | 400 |
| kermany | mahalanobis_l2 | null | 0 | +0.0001 | 0.0000 | 0.0008 | 400 |
| kermany | mahalanobis_l2 | d01 | 0.1247 | +0.0077 | 0.0001 | 0.0014 | 400 |
| kermany | mahalanobis_l2 | d03 | 0.2304 | +0.0263 | 0.0001 | 0.0024 | 400 |
| kermany | mahalanobis_l2 | d05 | 0.308 | +0.0460 | 0.0002 | 0.0036 | 400 |
| kermany | mahalanobis_l2 | moderate | 0.6519 | +0.1489 | 0.0005 | 0.0109 | 400 |

Targets not reachable at ICC <= 0.95 in the pilot (replaced by one 'max' level = pilot ICC with the largest mean Delta):

- dermamnist / mahalanobis_l2: d05, moderate
- dermamnist / knn_mean_cosine: moderate
- isic2019 / mahalanobis_l2: moderate

## Coverage per cell

| design | scorer | level | true Δ | method | n | coverage ± MC SE | width | SE/SD | lo>0.02 | lo>0 | s/call |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | +0.0024 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0197 | 1.57 | 0.00 | 0.02 | 56.97 |
| camelyon | knn_mean_cosine | null | +0.0024 | bootstrap old (OOD images) | 100 | 0.82 ± 0.04 | 0.0095 | 0.77 | 0.00 | 0.06 | 2.62 |
| camelyon | knn_mean_cosine | null | +0.0024 | bootstrap (OOD groups) | 100 | 0.82 ± 0.04 | 0.0095 | 0.77 | 0.00 | 0.06 | 2.61 |
| camelyon | knn_mean_cosine | d01 | +0.0093 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0234 | 1.68 | 0.00 | 0.31 | 57.08 |
| camelyon | knn_mean_cosine | d01 | +0.0093 | bootstrap old (OOD images) | 100 | 0.84 ± 0.04 | 0.0135 | 0.99 | 0.00 | 0.67 | 2.55 |
| camelyon | knn_mean_cosine | d01 | +0.0093 | bootstrap (OOD groups) | 100 | 0.84 ± 0.04 | 0.0135 | 0.99 | 0.00 | 0.67 | 2.56 |
| camelyon | knn_mean_cosine | d03 | +0.0249 | jackknife (default) | 100 | 0.97 ± 0.02 | 0.0344 | 1.80 | 0.00 | 0.90 | 54.27 |
| camelyon | knn_mean_cosine | d03 | +0.0249 | bootstrap old (OOD images) | 100 | 0.88 ± 0.03 | 0.0258 | 1.40 | 0.00 | 1.00 | 2.33 |
| camelyon | knn_mean_cosine | d03 | +0.0249 | bootstrap (OOD groups) | 100 | 0.88 ± 0.03 | 0.0258 | 1.40 | 0.00 | 1.00 | 2.38 |
| camelyon | knn_mean_cosine | d05 | +0.0476 | jackknife (default) | 100 | 0.94 ± 0.02 | 0.0494 | 1.82 | 0.68 | 1.00 | 51.89 |
| camelyon | knn_mean_cosine | d05 | +0.0476 | bootstrap old (OOD images) | 100 | 0.91 ± 0.03 | 0.0427 | 1.64 | 0.55 | 1.00 | 2.35 |
| camelyon | knn_mean_cosine | d05 | +0.0476 | bootstrap (OOD groups) | 100 | 0.91 ± 0.03 | 0.0427 | 1.64 | 0.55 | 1.00 | 2.32 |
| camelyon | knn_mean_cosine | moderate | +0.1416 | jackknife (default) | 100 | 0.92 ± 0.03 | 0.0852 | 1.56 | 1.00 | 1.00 | 51.86 |
| camelyon | knn_mean_cosine | moderate | +0.1416 | bootstrap old (OOD images) | 100 | 0.91 ± 0.03 | 0.0850 | 1.62 | 1.00 | 1.00 | 2.29 |
| camelyon | knn_mean_cosine | moderate | +0.1416 | bootstrap (OOD groups) | 100 | 0.91 ± 0.03 | 0.0850 | 1.62 | 1.00 | 1.00 | 2.31 |
| camelyon | mahalanobis_l2 | null | +0.0022 | jackknife (default) | 100 | 0.93 ± 0.03 | 0.0170 | 1.58 | 0.00 | 0.01 | 2.77 |
| camelyon | mahalanobis_l2 | null | +0.0022 | bootstrap old (OOD images) | 100 | 0.75 ± 0.04 | 0.0074 | 0.69 | 0.00 | 0.09 | 0.66 |
| camelyon | mahalanobis_l2 | null | +0.0022 | bootstrap (OOD groups) | 100 | 0.75 ± 0.04 | 0.0074 | 0.69 | 0.00 | 0.09 | 0.67 |
| camelyon | mahalanobis_l2 | d01 | +0.0093 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0203 | 1.72 | 0.00 | 0.55 | 2.65 |
| camelyon | mahalanobis_l2 | d01 | +0.0093 | bootstrap old (OOD images) | 100 | 0.85 ± 0.04 | 0.0118 | 1.03 | 0.00 | 0.92 | 0.68 |
| camelyon | mahalanobis_l2 | d01 | +0.0093 | bootstrap (OOD groups) | 100 | 0.85 ± 0.04 | 0.0118 | 1.03 | 0.00 | 0.92 | 0.67 |
| camelyon | mahalanobis_l2 | d03 | +0.0245 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0297 | 1.89 | 0.01 | 1.00 | 2.61 |
| camelyon | mahalanobis_l2 | d03 | +0.0245 | bootstrap old (OOD images) | 100 | 0.92 ± 0.03 | 0.0219 | 1.46 | 0.00 | 1.00 | 0.63 |
| camelyon | mahalanobis_l2 | d03 | +0.0245 | bootstrap (OOD groups) | 100 | 0.92 ± 0.03 | 0.0219 | 1.46 | 0.00 | 1.00 | 0.62 |
| camelyon | mahalanobis_l2 | d05 | +0.0448 | jackknife (default) | 100 | 0.95 ± 0.02 | 0.0402 | 1.90 | 0.80 | 1.00 | 2.49 |
| camelyon | mahalanobis_l2 | d05 | +0.0448 | bootstrap old (OOD images) | 100 | 0.90 ± 0.03 | 0.0324 | 1.59 | 0.91 | 1.00 | 0.59 |
| camelyon | mahalanobis_l2 | d05 | +0.0448 | bootstrap (OOD groups) | 100 | 0.90 ± 0.03 | 0.0324 | 1.59 | 0.91 | 1.00 | 0.58 |
| camelyon | mahalanobis_l2 | moderate | +0.1445 | jackknife (default) | 100 | 0.93 ± 0.03 | 0.0613 | 1.41 | 1.00 | 1.00 | 2.55 |
| camelyon | mahalanobis_l2 | moderate | +0.1445 | bootstrap old (OOD images) | 100 | 0.91 ± 0.03 | 0.0588 | 1.40 | 1.00 | 1.00 | 0.60 |
| camelyon | mahalanobis_l2 | moderate | +0.1445 | bootstrap (OOD groups) | 100 | 0.91 ± 0.03 | 0.0588 | 1.40 | 1.00 | 1.00 | 0.60 |
| breakhis | knn_mean_cosine | null | +0.0006 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0105 | 1.39 | 0.00 | 0.02 | 15.23 |
| breakhis | knn_mean_cosine | null | +0.0006 | bootstrap old (OOD images) | 100 | 0.90 ± 0.03 | 0.0068 | 0.90 | 0.00 | 0.12 | 0.49 |
| breakhis | knn_mean_cosine | null | +0.0006 | bootstrap (OOD groups) | 100 | 0.90 ± 0.03 | 0.0068 | 0.90 | 0.00 | 0.12 | 0.49 |
| breakhis | knn_mean_cosine | d01 | +0.0080 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0120 | 1.38 | 0.00 | 0.82 | 15.50 |
| breakhis | knn_mean_cosine | d01 | +0.0080 | bootstrap old (OOD images) | 100 | 0.90 ± 0.03 | 0.0079 | 0.91 | 0.00 | 0.99 | 0.48 |
| breakhis | knn_mean_cosine | d01 | +0.0080 | bootstrap (OOD groups) | 100 | 0.90 ± 0.03 | 0.0079 | 0.91 | 0.00 | 0.99 | 0.49 |
| breakhis | knn_mean_cosine | d03 | +0.0243 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0153 | 1.31 | 0.16 | 1.00 | 15.53 |
| breakhis | knn_mean_cosine | d03 | +0.0243 | bootstrap old (OOD images) | 100 | 0.94 ± 0.02 | 0.0111 | 0.96 | 0.35 | 1.00 | 0.49 |
| breakhis | knn_mean_cosine | d03 | +0.0243 | bootstrap (OOD groups) | 100 | 0.94 ± 0.02 | 0.0111 | 0.96 | 0.35 | 1.00 | 0.49 |
| breakhis | knn_mean_cosine | d05 | +0.0485 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0198 | 1.25 | 1.00 | 1.00 | 14.56 |
| breakhis | knn_mean_cosine | d05 | +0.0485 | bootstrap old (OOD images) | 100 | 0.91 ± 0.03 | 0.0163 | 1.03 | 1.00 | 1.00 | 0.46 |
| breakhis | knn_mean_cosine | d05 | +0.0485 | bootstrap (OOD groups) | 100 | 0.91 ± 0.03 | 0.0163 | 1.03 | 1.00 | 1.00 | 0.45 |
| breakhis | knn_mean_cosine | moderate | +0.1290 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0359 | 1.14 | 1.00 | 1.00 | 14.40 |
| breakhis | knn_mean_cosine | moderate | +0.1290 | bootstrap old (OOD images) | 100 | 0.93 ± 0.03 | 0.0317 | 1.01 | 1.00 | 1.00 | 0.43 |
| breakhis | knn_mean_cosine | moderate | +0.1290 | bootstrap (OOD groups) | 100 | 0.93 ± 0.03 | 0.0317 | 1.01 | 1.00 | 1.00 | 0.43 |
| breakhis | mahalanobis_l2 | null | +0.0007 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0062 | 1.40 | 0.00 | 0.00 | 1.80 |
| breakhis | mahalanobis_l2 | null | +0.0007 | bootstrap old (OOD images) | 100 | 0.88 ± 0.03 | 0.0035 | 0.78 | 0.00 | 0.09 | 0.19 |
| breakhis | mahalanobis_l2 | null | +0.0007 | bootstrap (OOD groups) | 100 | 0.88 ± 0.03 | 0.0035 | 0.78 | 0.00 | 0.09 | 0.19 |
| breakhis | mahalanobis_l2 | d01 | +0.0079 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0088 | 1.42 | 0.00 | 0.99 | 1.91 |
| breakhis | mahalanobis_l2 | d01 | +0.0079 | bootstrap old (OOD images) | 100 | 0.88 ± 0.03 | 0.0056 | 0.89 | 0.00 | 1.00 | 0.20 |
| breakhis | mahalanobis_l2 | d01 | +0.0079 | bootstrap (OOD groups) | 100 | 0.88 ± 0.03 | 0.0056 | 0.89 | 0.00 | 1.00 | 0.21 |
| breakhis | mahalanobis_l2 | d03 | +0.0238 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0137 | 1.30 | 0.09 | 1.00 | 1.69 |
| breakhis | mahalanobis_l2 | d03 | +0.0238 | bootstrap old (OOD images) | 100 | 0.90 ± 0.03 | 0.0096 | 0.92 | 0.27 | 1.00 | 0.17 |
| breakhis | mahalanobis_l2 | d03 | +0.0238 | bootstrap (OOD groups) | 100 | 0.90 ± 0.03 | 0.0096 | 0.92 | 0.27 | 1.00 | 0.17 |
| breakhis | mahalanobis_l2 | d05 | +0.0444 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0189 | 1.26 | 1.00 | 1.00 | 1.67 |
| breakhis | mahalanobis_l2 | d05 | +0.0444 | bootstrap old (OOD images) | 100 | 0.93 ± 0.03 | 0.0143 | 0.95 | 1.00 | 1.00 | 0.18 |
| breakhis | mahalanobis_l2 | d05 | +0.0444 | bootstrap (OOD groups) | 100 | 0.93 ± 0.03 | 0.0143 | 0.95 | 1.00 | 1.00 | 0.18 |
| breakhis | mahalanobis_l2 | moderate | +0.1367 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0370 | 1.15 | 1.00 | 1.00 | 1.70 |
| breakhis | mahalanobis_l2 | moderate | +0.1367 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0313 | 0.98 | 1.00 | 1.00 | 0.17 |
| breakhis | mahalanobis_l2 | moderate | +0.1367 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0313 | 0.98 | 1.00 | 1.00 | 0.17 |
| dermamnist | knn_mean_cosine | null | +0.0001 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0098 | 1.38 | 0.00 | 0.00 | 48.33 |
| dermamnist | knn_mean_cosine | null | +0.0001 | bootstrap old (OOD images) | 100 | 0.96 ± 0.02 | 0.0070 | 0.98 | 0.00 | 0.00 | 1.48 |
| dermamnist | knn_mean_cosine | null | +0.0001 | bootstrap (OOD groups) | 100 | 0.96 ± 0.02 | 0.0070 | 0.98 | 0.00 | 0.00 | 1.48 |
| dermamnist | knn_mean_cosine | d01 | +0.0079 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0098 | 1.41 | 0.00 | 0.90 | 48.46 |
| dermamnist | knn_mean_cosine | d01 | +0.0079 | bootstrap old (OOD images) | 100 | 0.95 ± 0.02 | 0.0071 | 1.03 | 0.00 | 0.98 | 1.51 |
| dermamnist | knn_mean_cosine | d01 | +0.0079 | bootstrap (OOD groups) | 100 | 0.95 ± 0.02 | 0.0071 | 1.03 | 0.00 | 0.98 | 1.52 |
| dermamnist | knn_mean_cosine | d03 | +0.0297 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0105 | 1.44 | 0.99 | 1.00 | 47.61 |
| dermamnist | knn_mean_cosine | d03 | +0.0297 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0079 | 1.08 | 0.99 | 1.00 | 1.38 |
| dermamnist | knn_mean_cosine | d03 | +0.0297 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0079 | 1.08 | 0.99 | 1.00 | 1.45 |
| dermamnist | knn_mean_cosine | d05 | +0.0496 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0118 | 1.45 | 1.00 | 1.00 | 49.93 |
| dermamnist | knn_mean_cosine | d05 | +0.0496 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0091 | 1.12 | 1.00 | 1.00 | 1.56 |
| dermamnist | knn_mean_cosine | d05 | +0.0496 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0091 | 1.12 | 1.00 | 1.00 | 1.56 |
| dermamnist | knn_mean_cosine | max | +0.0887 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0156 | 1.24 | 1.00 | 1.00 | 48.56 |
| dermamnist | knn_mean_cosine | max | +0.0887 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0128 | 1.02 | 1.00 | 1.00 | 1.50 |
| dermamnist | knn_mean_cosine | max | +0.0887 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0128 | 1.02 | 1.00 | 1.00 | 1.48 |
| dermamnist | mahalanobis_l2 | null | +0.0001 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0039 | 1.42 | 0.00 | 0.01 | 3.26 |
| dermamnist | mahalanobis_l2 | null | +0.0001 | bootstrap old (OOD images) | 100 | 0.94 ± 0.02 | 0.0027 | 1.00 | 0.00 | 0.01 | 0.45 |
| dermamnist | mahalanobis_l2 | null | +0.0001 | bootstrap (OOD groups) | 100 | 0.94 ± 0.02 | 0.0027 | 1.00 | 0.00 | 0.01 | 0.45 |
| dermamnist | mahalanobis_l2 | d01 | +0.0093 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0058 | 1.45 | 0.00 | 1.00 | 3.33 |
| dermamnist | mahalanobis_l2 | d01 | +0.0093 | bootstrap old (OOD images) | 100 | 0.93 ± 0.03 | 0.0040 | 1.00 | 0.00 | 1.00 | 0.43 |
| dermamnist | mahalanobis_l2 | d01 | +0.0093 | bootstrap (OOD groups) | 100 | 0.93 ± 0.03 | 0.0040 | 1.00 | 0.00 | 1.00 | 0.44 |
| dermamnist | mahalanobis_l2 | d03 | +0.0298 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0097 | 1.39 | 1.00 | 1.00 | 3.00 |
| dermamnist | mahalanobis_l2 | d03 | +0.0298 | bootstrap old (OOD images) | 100 | 0.96 ± 0.02 | 0.0065 | 0.93 | 1.00 | 1.00 | 0.43 |
| dermamnist | mahalanobis_l2 | d03 | +0.0298 | bootstrap (OOD groups) | 100 | 0.96 ± 0.02 | 0.0065 | 0.93 | 1.00 | 1.00 | 0.42 |
| dermamnist | mahalanobis_l2 | max | +0.0364 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0109 | 1.37 | 1.00 | 1.00 | 3.20 |
| dermamnist | mahalanobis_l2 | max | +0.0364 | bootstrap old (OOD images) | 100 | 0.95 ± 0.02 | 0.0073 | 0.92 | 1.00 | 1.00 | 0.49 |
| dermamnist | mahalanobis_l2 | max | +0.0364 | bootstrap (OOD groups) | 100 | 0.95 ± 0.02 | 0.0073 | 0.92 | 1.00 | 1.00 | 0.50 |
| isic2019 | knn_mean_cosine | null | +0.0000 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0048 | 1.55 | 0.00 | 0.01 | 135.79 |
| isic2019 | knn_mean_cosine | null | +0.0000 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0033 | 1.07 | 0.00 | 0.03 | 3.14 |
| isic2019 | knn_mean_cosine | null | +0.0000 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0033 | 1.07 | 0.00 | 0.03 | 3.16 |
| isic2019 | knn_mean_cosine | d01 | +0.0072 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0050 | 1.43 | 0.00 | 1.00 | 135.57 |
| isic2019 | knn_mean_cosine | d01 | +0.0072 | bootstrap old (OOD images) | 100 | 0.92 ± 0.03 | 0.0036 | 1.03 | 0.00 | 1.00 | 3.28 |
| isic2019 | knn_mean_cosine | d01 | +0.0072 | bootstrap (OOD groups) | 100 | 0.92 ± 0.03 | 0.0036 | 1.03 | 0.00 | 1.00 | 3.14 |
| isic2019 | knn_mean_cosine | d03 | +0.0276 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0067 | 1.43 | 1.00 | 1.00 | 134.75 |
| isic2019 | knn_mean_cosine | d03 | +0.0276 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0057 | 1.22 | 1.00 | 1.00 | 3.15 |
| isic2019 | knn_mean_cosine | d03 | +0.0276 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0057 | 1.22 | 1.00 | 1.00 | 3.06 |
| isic2019 | knn_mean_cosine | d05 | +0.0477 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0090 | 1.43 | 1.00 | 1.00 | 137.31 |
| isic2019 | knn_mean_cosine | d05 | +0.0477 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0083 | 1.32 | 1.00 | 1.00 | 3.24 |
| isic2019 | knn_mean_cosine | d05 | +0.0477 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0083 | 1.32 | 1.00 | 1.00 | 3.19 |
| isic2019 | knn_mean_cosine | moderate | +0.1483 | jackknife (default) | 100 | 0.96 ± 0.02 | 0.0230 | 1.26 | 1.00 | 1.00 | 135.28 |
| isic2019 | knn_mean_cosine | moderate | +0.1483 | bootstrap old (OOD images) | 100 | 0.96 ± 0.02 | 0.0218 | 1.19 | 1.00 | 1.00 | 3.19 |
| isic2019 | knn_mean_cosine | moderate | +0.1483 | bootstrap (OOD groups) | 100 | 0.96 ± 0.02 | 0.0218 | 1.19 | 1.00 | 1.00 | 3.17 |
| isic2019 | mahalanobis_l2 | null | +0.0000 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0024 | 1.64 | 0.00 | 0.00 | 6.87 |
| isic2019 | mahalanobis_l2 | null | +0.0000 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0016 | 1.10 | 0.00 | 0.02 | 0.46 |
| isic2019 | mahalanobis_l2 | null | +0.0000 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0016 | 1.10 | 0.00 | 0.02 | 0.46 |
| isic2019 | mahalanobis_l2 | d01 | +0.0092 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0040 | 1.49 | 0.00 | 1.00 | 7.07 |
| isic2019 | mahalanobis_l2 | d01 | +0.0092 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0028 | 1.06 | 0.00 | 1.00 | 0.45 |
| isic2019 | mahalanobis_l2 | d01 | +0.0092 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0028 | 1.06 | 0.00 | 1.00 | 0.46 |
| isic2019 | mahalanobis_l2 | d03 | +0.0294 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0078 | 1.57 | 1.00 | 1.00 | 7.22 |
| isic2019 | mahalanobis_l2 | d03 | +0.0294 | bootstrap old (OOD images) | 100 | 0.96 ± 0.02 | 0.0060 | 1.21 | 1.00 | 1.00 | 0.47 |
| isic2019 | mahalanobis_l2 | d03 | +0.0294 | bootstrap (OOD groups) | 100 | 0.96 ± 0.02 | 0.0060 | 1.21 | 1.00 | 1.00 | 0.46 |
| isic2019 | mahalanobis_l2 | d05 | +0.0493 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0116 | 1.54 | 1.00 | 1.00 | 7.44 |
| isic2019 | mahalanobis_l2 | d05 | +0.0493 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0093 | 1.23 | 1.00 | 1.00 | 0.49 |
| isic2019 | mahalanobis_l2 | d05 | +0.0493 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0093 | 1.23 | 1.00 | 1.00 | 0.49 |
| isic2019 | mahalanobis_l2 | max | +0.0563 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0129 | 1.51 | 1.00 | 1.00 | 7.38 |
| isic2019 | mahalanobis_l2 | max | +0.0563 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0104 | 1.23 | 1.00 | 1.00 | 0.49 |
| isic2019 | mahalanobis_l2 | max | +0.0563 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0104 | 1.23 | 1.00 | 1.00 | 0.49 |
| kermany | knn_mean_cosine | null | -0.0000 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0092 | 1.51 | 0.00 | 0.00 | 140.04 |
| kermany | knn_mean_cosine | null | -0.0000 | bootstrap old (OOD images) | 100 | 0.94 ± 0.02 | 0.0064 | 1.05 | 0.00 | 0.02 | 2.85 |
| kermany | knn_mean_cosine | null | -0.0000 | bootstrap (OOD groups) | 100 | 0.94 ± 0.02 | 0.0064 | 1.05 | 0.00 | 0.02 | 2.87 |
| kermany | knn_mean_cosine | d01 | +0.0066 | jackknife (default) | 100 | 0.97 ± 0.02 | 0.0102 | 1.53 | 0.00 | 0.79 | 142.12 |
| kermany | knn_mean_cosine | d01 | +0.0066 | bootstrap old (OOD images) | 100 | 0.95 ± 0.02 | 0.0075 | 1.13 | 0.00 | 0.97 | 3.01 |
| kermany | knn_mean_cosine | d01 | +0.0066 | bootstrap (OOD groups) | 100 | 0.95 ± 0.02 | 0.0075 | 1.13 | 0.00 | 0.97 | 3.04 |
| kermany | knn_mean_cosine | d03 | +0.0219 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0144 | 1.63 | 0.01 | 1.00 | 144.60 |
| kermany | knn_mean_cosine | d03 | +0.0219 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0123 | 1.40 | 0.02 | 1.00 | 3.11 |
| kermany | knn_mean_cosine | d03 | +0.0219 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0123 | 1.40 | 0.02 | 1.00 | 3.16 |
| kermany | knn_mean_cosine | d05 | +0.0450 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0219 | 1.68 | 1.00 | 1.00 | 144.29 |
| kermany | knn_mean_cosine | d05 | +0.0450 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0203 | 1.57 | 1.00 | 1.00 | 3.08 |
| kermany | knn_mean_cosine | d05 | +0.0450 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0203 | 1.57 | 1.00 | 1.00 | 3.20 |
| kermany | knn_mean_cosine | moderate | +0.1483 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0466 | 1.50 | 1.00 | 1.00 | 136.33 |
| kermany | knn_mean_cosine | moderate | +0.1483 | bootstrap old (OOD images) | 100 | 0.98 ± 0.01 | 0.0444 | 1.44 | 1.00 | 1.00 | 2.61 |
| kermany | knn_mean_cosine | moderate | +0.1483 | bootstrap (OOD groups) | 100 | 0.98 ± 0.01 | 0.0444 | 1.44 | 1.00 | 1.00 | 2.63 |
| kermany | mahalanobis_l2 | null | +0.0001 | jackknife (default) | 100 | 0.98 ± 0.01 | 0.0045 | 1.36 | 0.00 | 0.00 | 14.76 |
| kermany | mahalanobis_l2 | null | +0.0001 | bootstrap old (OOD images) | 100 | 0.96 ± 0.02 | 0.0032 | 0.95 | 0.00 | 0.01 | 0.45 |
| kermany | mahalanobis_l2 | null | +0.0001 | bootstrap (OOD groups) | 100 | 0.96 ± 0.02 | 0.0032 | 0.95 | 0.00 | 0.01 | 0.45 |
| kermany | mahalanobis_l2 | d01 | +0.0077 | jackknife (default) | 100 | 1.00 ± 0.00 | 0.0087 | 1.63 | 0.00 | 0.99 | 14.23 |
| kermany | mahalanobis_l2 | d01 | +0.0077 | bootstrap old (OOD images) | 100 | 0.97 ± 0.02 | 0.0064 | 1.20 | 0.00 | 1.00 | 0.44 |
| kermany | mahalanobis_l2 | d01 | +0.0077 | bootstrap (OOD groups) | 100 | 0.97 ± 0.02 | 0.0064 | 1.20 | 0.00 | 1.00 | 0.44 |
| kermany | mahalanobis_l2 | d03 | +0.0263 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0171 | 1.78 | 0.17 | 1.00 | 14.73 |
| kermany | mahalanobis_l2 | d03 | +0.0263 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0140 | 1.46 | 0.35 | 1.00 | 0.46 |
| kermany | mahalanobis_l2 | d03 | +0.0263 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0140 | 1.46 | 0.35 | 1.00 | 0.47 |
| kermany | mahalanobis_l2 | d05 | +0.0460 | jackknife (default) | 100 | 0.99 ± 0.01 | 0.0251 | 1.79 | 1.00 | 1.00 | 15.20 |
| kermany | mahalanobis_l2 | d05 | +0.0460 | bootstrap old (OOD images) | 100 | 0.99 ± 0.01 | 0.0216 | 1.54 | 1.00 | 1.00 | 0.47 |
| kermany | mahalanobis_l2 | d05 | +0.0460 | bootstrap (OOD groups) | 100 | 0.99 ± 0.01 | 0.0216 | 1.54 | 1.00 | 1.00 | 0.48 |
| kermany | mahalanobis_l2 | moderate | +0.1489 | jackknife (default) | 100 | 0.97 ± 0.02 | 0.0594 | 1.39 | 1.00 | 1.00 | 15.03 |
| kermany | mahalanobis_l2 | moderate | +0.1489 | bootstrap old (OOD images) | 100 | 0.96 ± 0.02 | 0.0533 | 1.25 | 1.00 | 1.00 | 0.48 |
| kermany | mahalanobis_l2 | moderate | +0.1489 | bootstrap (OOD groups) | 100 | 0.96 ± 0.02 | 0.0533 | 1.25 | 1.00 | 1.00 | 0.48 |

## Compact: coverage by method (rows = cells)

Cells: n datasets (jackknife) in brackets; MC SE of 0.95 is 0.022 at n=100, 0.031 at n=50.

| design | scorer | level | true Δ | jackknife [n] | bootstrap old | bootstrap OOD-groups | subsample-normal [n] |
|---|---|---|---:|---:|---:|---:|---:|
| camelyon | knn_mean_cosine | null | +0.0024 | 0.98 ± 0.01 [100] | 0.82 ± 0.04 | 0.82 ± 0.04 | — |
| camelyon | knn_mean_cosine | d01 | +0.0093 | 0.98 ± 0.01 [100] | 0.84 ± 0.04 | 0.84 ± 0.04 | — |
| camelyon | knn_mean_cosine | d03 | +0.0249 | 0.97 ± 0.02 [100] | 0.88 ± 0.03 | 0.88 ± 0.03 | — |
| camelyon | knn_mean_cosine | d05 | +0.0476 | 0.94 ± 0.02 [100] | 0.91 ± 0.03 | 0.91 ± 0.03 | — |
| camelyon | knn_mean_cosine | moderate | +0.1416 | 0.92 ± 0.03 [100] | 0.91 ± 0.03 | 0.91 ± 0.03 | — |
| camelyon | mahalanobis_l2 | null | +0.0022 | 0.93 ± 0.03 [100] | 0.75 ± 0.04 | 0.75 ± 0.04 | — |
| camelyon | mahalanobis_l2 | d01 | +0.0093 | 0.96 ± 0.02 [100] | 0.85 ± 0.04 | 0.85 ± 0.04 | — |
| camelyon | mahalanobis_l2 | d03 | +0.0245 | 0.96 ± 0.02 [100] | 0.92 ± 0.03 | 0.92 ± 0.03 | — |
| camelyon | mahalanobis_l2 | d05 | +0.0448 | 0.95 ± 0.02 [100] | 0.90 ± 0.03 | 0.90 ± 0.03 | — |
| camelyon | mahalanobis_l2 | moderate | +0.1445 | 0.93 ± 0.03 [100] | 0.91 ± 0.03 | 0.91 ± 0.03 | — |
| breakhis | knn_mean_cosine | null | +0.0006 | 0.99 ± 0.01 [100] | 0.90 ± 0.03 | 0.90 ± 0.03 | — |
| breakhis | knn_mean_cosine | d01 | +0.0080 | 0.99 ± 0.01 [100] | 0.90 ± 0.03 | 0.90 ± 0.03 | — |
| breakhis | knn_mean_cosine | d03 | +0.0243 | 0.98 ± 0.01 [100] | 0.94 ± 0.02 | 0.94 ± 0.02 | — |
| breakhis | knn_mean_cosine | d05 | +0.0485 | 0.98 ± 0.01 [100] | 0.91 ± 0.03 | 0.91 ± 0.03 | — |
| breakhis | knn_mean_cosine | moderate | +0.1290 | 0.98 ± 0.01 [100] | 0.93 ± 0.03 | 0.93 ± 0.03 | — |
| breakhis | mahalanobis_l2 | null | +0.0007 | 1.00 ± 0.00 [100] | 0.88 ± 0.03 | 0.88 ± 0.03 | — |
| breakhis | mahalanobis_l2 | d01 | +0.0079 | 0.99 ± 0.01 [100] | 0.88 ± 0.03 | 0.88 ± 0.03 | — |
| breakhis | mahalanobis_l2 | d03 | +0.0238 | 0.99 ± 0.01 [100] | 0.90 ± 0.03 | 0.90 ± 0.03 | — |
| breakhis | mahalanobis_l2 | d05 | +0.0444 | 0.99 ± 0.01 [100] | 0.93 ± 0.03 | 0.93 ± 0.03 | — |
| breakhis | mahalanobis_l2 | moderate | +0.1367 | 0.99 ± 0.01 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| dermamnist | knn_mean_cosine | null | +0.0001 | 0.98 ± 0.01 [100] | 0.96 ± 0.02 | 0.96 ± 0.02 | — |
| dermamnist | knn_mean_cosine | d01 | +0.0079 | 0.99 ± 0.01 [100] | 0.95 ± 0.02 | 0.95 ± 0.02 | — |
| dermamnist | knn_mean_cosine | d03 | +0.0297 | 0.99 ± 0.01 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| dermamnist | knn_mean_cosine | d05 | +0.0496 | 0.99 ± 0.01 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| dermamnist | knn_mean_cosine | max | +0.0887 | 1.00 ± 0.00 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| dermamnist | mahalanobis_l2 | null | +0.0001 | 1.00 ± 0.00 [100] | 0.94 ± 0.02 | 0.94 ± 0.02 | — |
| dermamnist | mahalanobis_l2 | d01 | +0.0093 | 0.98 ± 0.01 [100] | 0.93 ± 0.03 | 0.93 ± 0.03 | — |
| dermamnist | mahalanobis_l2 | d03 | +0.0298 | 0.99 ± 0.01 [100] | 0.96 ± 0.02 | 0.96 ± 0.02 | — |
| dermamnist | mahalanobis_l2 | max | +0.0364 | 1.00 ± 0.00 [100] | 0.95 ± 0.02 | 0.95 ± 0.02 | — |
| isic2019 | knn_mean_cosine | null | +0.0000 | 0.99 ± 0.01 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| isic2019 | knn_mean_cosine | d01 | +0.0072 | 0.99 ± 0.01 [100] | 0.92 ± 0.03 | 0.92 ± 0.03 | — |
| isic2019 | knn_mean_cosine | d03 | +0.0276 | 1.00 ± 0.00 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| isic2019 | knn_mean_cosine | d05 | +0.0477 | 1.00 ± 0.00 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| isic2019 | knn_mean_cosine | moderate | +0.1483 | 0.96 ± 0.02 [100] | 0.96 ± 0.02 | 0.96 ± 0.02 | — |
| isic2019 | mahalanobis_l2 | null | +0.0000 | 1.00 ± 0.00 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| isic2019 | mahalanobis_l2 | d01 | +0.0092 | 1.00 ± 0.00 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| isic2019 | mahalanobis_l2 | d03 | +0.0294 | 1.00 ± 0.00 [100] | 0.96 ± 0.02 | 0.96 ± 0.02 | — |
| isic2019 | mahalanobis_l2 | d05 | +0.0493 | 1.00 ± 0.00 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| isic2019 | mahalanobis_l2 | max | +0.0563 | 1.00 ± 0.00 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| kermany | knn_mean_cosine | null | -0.0000 | 1.00 ± 0.00 [100] | 0.94 ± 0.02 | 0.94 ± 0.02 | — |
| kermany | knn_mean_cosine | d01 | +0.0066 | 0.97 ± 0.02 [100] | 0.95 ± 0.02 | 0.95 ± 0.02 | — |
| kermany | knn_mean_cosine | d03 | +0.0219 | 1.00 ± 0.00 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| kermany | knn_mean_cosine | d05 | +0.0450 | 1.00 ± 0.00 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| kermany | knn_mean_cosine | moderate | +0.1483 | 0.99 ± 0.01 [100] | 0.98 ± 0.01 | 0.98 ± 0.01 | — |
| kermany | mahalanobis_l2 | null | +0.0001 | 0.98 ± 0.01 [100] | 0.96 ± 0.02 | 0.96 ± 0.02 | — |
| kermany | mahalanobis_l2 | d01 | +0.0077 | 1.00 ± 0.00 [100] | 0.97 ± 0.02 | 0.97 ± 0.02 | — |
| kermany | mahalanobis_l2 | d03 | +0.0263 | 0.99 ± 0.01 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| kermany | mahalanobis_l2 | d05 | +0.0460 | 0.99 ± 0.01 [100] | 0.99 ± 0.01 | 0.99 ± 0.01 | — |
| kermany | mahalanobis_l2 | moderate | +0.1489 | 0.97 ± 0.02 [100] | 0.96 ± 0.02 | 0.96 ± 0.02 | — |

## Range per scorer x method (all designs and levels)

| scorer | method | coverage min–max | SE/SD min–max | cells |
|---|---|---|---|---:|

## Pre-registered decision (jackknife, threshold 0.93, fixed before results)

Cells: 49; jackknife coverage min = 0.92; cells < 0.93: 1.

**Outcome: at least one cell < 0.93 -> reported plainly; no method changes (stop).**

- camelyon / knn_mean_cosine / moderate: 0.92
