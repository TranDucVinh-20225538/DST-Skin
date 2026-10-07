# Fixes (F)

Reference F1 = group-disjoint ID (2-fold disjoint AUROC). Per dataset: median |AUROC_fix − AUROC_F1| over
the cells the fix changes, median Kendall tau-b (fix 7-score vector vs F1) over archs, median change vs the
standard protocol. Bar: median |diff| <= 0.02 AND median tau >= 0.8 on >= 2/3 of datasets.

| fix | dataset | median |diff| vs F1 | median tau | median change vs standard | archs | works |
|---|---|---|---|---|---|---|
| F2 | camelyon | 0.016 | 0.81 | -0.063 | 8 | yes |
| F2 | iwildcam | 0.030 | 0.90 | -0.106 | 8 | no |
| F2 | rxrx1 | 0.000 | 1.00 | -0.025 | 4 | yes |
| F3a | camelyon | 0.079 | 0.43 | -0.137 | 8 | no |
| F3a | iwildcam | 0.068 | 0.33 | -0.175 | 8 | no |
| F3a | rxrx1 | 0.053 | 0.24 | +0.016 | 4 | no |
| F3b | camelyon | 0.134 | 0.71 | +0.050 | 8 | no |
| F3b | iwildcam | 0.156 | -0.10 | -0.018 | 8 | no |
| F3b | rxrx1 | 0.062 | 0.33 | +0.001 | 4 | no |
| F3c | camelyon | 0.108 | 0.62 | -0.099 | 4 | no |
| F3c | iwildcam | 0.145 | -0.05 | +0.006 | 4 | no |
| F3c | rxrx1 | 0.038 | 0.57 | -0.042 | 4 | no |
| F4 | camelyon | 0.011 | 0.86 | -0.097 | 8 | yes |
| F4 | iwildcam | 0.012 | 0.29 | -0.216 | 8 | no |
| F4 | rxrx1 | 0.028 | 0.81 | -0.064 | 4 | no |

Verdicts: F2 works (2/3); F3a does not work (0/3); F3b does not work (0/3); F3c does not work (0/3); F4 does not work (1/3)

F2 = cross-fitted scorer (full ID kept); F3a = per-group centring; F3b = per-batch centring (256);
F3c = Macenko / colour-standardised features (4 archs); F4 = kNN with same-group exclusion (kNN only).
