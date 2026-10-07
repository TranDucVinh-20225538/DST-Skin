# R3 item 4: anisotropic synthetic test

commit: 9b39bcd

Design and predictions: PRECOMMIT.json (committed before any run). Protocol paper_2fold, K = 2, n_groups_fit = 15 slides per fold (30 training slides), d in {768, 2560}, n/d = mean per-fold fit size / d. Model ID accuracy: not applicable (synthetic features, no classifier); probe column = slide-ID probe balanced accuracy at the calibrated sigma_b.

## Calibration

| d | placement | sigma_b | probe bacc | delta (OOD shift) |
|---|---|---|---|---|
| 768 | high | 1.0938 | 0.896 | 4.1875 |
| 768 | low | 1.4375 | 0.897 | 4.1875 |
| 2560 | high | 1.0625 | 0.904 | 5.1250 |
| 2560 | low | 2.2500 | 0.896 | 5.1250 |

## Delta per cell (mean over replicates)

| d | n/d | n_fit/fold | placement | scorer | n_rep | Delta | SE | AUROC seen | AUROC unseen |
|---|---|---|---|---|---|---|---|---|---|
| 768 | 1 | 768 | high | exact_maha | 50 | +0.0284 | 0.0042 | 0.6490 | 0.6205 |
| 768 | 1 | 768 | high | knn | 50 | +0.1898 | 0.0060 | 0.8358 | 0.6460 |
| 768 | 1 | 768 | high | lw_maha | 50 | +0.0526 | 0.0033 | 0.7277 | 0.6752 |
| 768 | 1 | 768 | high | vim | 50 | +0.0307 | 0.0040 | 0.7108 | 0.6801 |
| 768 | 1 | 768 | low | exact_maha | 50 | +0.0354 | 0.0041 | 0.6455 | 0.6101 |
| 768 | 1 | 768 | low | knn | 50 | +0.0040 | 0.0007 | 0.5101 | 0.5061 |
| 768 | 1 | 768 | low | lw_maha | 50 | +0.0464 | 0.0032 | 0.7304 | 0.6841 |
| 768 | 1 | 768 | low | vim | 50 | +0.0453 | 0.0042 | 0.7039 | 0.6586 |
| 768 | 10 | 7680 | high | exact_maha | 50 | +0.0430 | 0.0012 | 0.8368 | 0.7937 |
| 768 | 10 | 7680 | high | knn | 50 | +0.1646 | 0.0044 | 0.8558 | 0.6912 |
| 768 | 10 | 7680 | high | lw_maha | 50 | +0.0438 | 0.0012 | 0.8359 | 0.7921 |
| 768 | 10 | 7680 | high | vim | 50 | +0.0024 | 0.0007 | 0.8141 | 0.8117 |
| 768 | 10 | 7680 | low | exact_maha | 50 | +0.0776 | 0.0020 | 0.8286 | 0.7510 |
| 768 | 10 | 7680 | low | knn | 50 | +0.0032 | 0.0002 | 0.5137 | 0.5105 |
| 768 | 10 | 7680 | low | lw_maha | 50 | +0.0767 | 0.0020 | 0.8275 | 0.7508 |
| 768 | 10 | 7680 | low | vim | 50 | +0.1008 | 0.0025 | 0.8066 | 0.7058 |
| 2560 | 1 | 2560 | high | exact_maha | 50 | +0.0164 | 0.0020 | 0.6539 | 0.6375 |
| 2560 | 1 | 2560 | high | knn | 50 | +0.1522 | 0.0039 | 0.8343 | 0.6821 |
| 2560 | 1 | 2560 | high | lw_maha | 50 | +0.0260 | 0.0017 | 0.7180 | 0.6920 |
| 2560 | 1 | 2560 | high | vim | 50 | +0.0242 | 0.0014 | 0.7518 | 0.7276 |
| 2560 | 1 | 2560 | low | exact_maha | 50 | +0.0462 | 0.0025 | 0.6486 | 0.6024 |
| 2560 | 1 | 2560 | low | knn | 50 | +0.0023 | 0.0003 | 0.5052 | 0.5029 |
| 2560 | 1 | 2560 | low | lw_maha | 50 | +0.0645 | 0.0020 | 0.7242 | 0.6597 |
| 2560 | 1 | 2560 | low | vim | 50 | +0.0665 | 0.0020 | 0.7451 | 0.6786 |
| 2560 | 10 | 25598 | high | exact_maha | 50 | +0.0213 | 0.0006 | 0.8435 | 0.8222 |
| 2560 | 10 | 25598 | high | knn | 50 | +0.1438 | 0.0033 | 0.8443 | 0.7005 |
| 2560 | 10 | 25598 | high | lw_maha | 50 | +0.0215 | 0.0006 | 0.8432 | 0.8217 |
| 2560 | 10 | 25598 | high | vim | 50 | +0.0016 | 0.0003 | 0.8433 | 0.8416 |
| 2560 | 10 | 25598 | low | exact_maha | 50 | +0.0964 | 0.0026 | 0.8382 | 0.7417 |
| 2560 | 10 | 25598 | low | knn | 50 | +0.0021 | 0.0001 | 0.5059 | 0.5038 |
| 2560 | 10 | 25598 | low | lw_maha | 50 | +0.0961 | 0.0026 | 0.8376 | 0.7415 |
| 2560 | 10 | 25598 | low | vim | 50 | +0.0899 | 0.0024 | 0.8392 | 0.7492 |

## Contrast D = Delta_high - Delta_low (paired over seeds, 95% t-CI) and verdict

| d | n/d | scorer | prediction | D | 95% CI | verdict |
|---|---|---|---|---|---|---|
| 768 | 1 | exact_maha | Delta_high ~= Delta_low (|D| <= 0.01 and the 95% CI of D inside [-0.02, 0.02]) | -0.0070 | [-0.0120, -0.0019] | hold |
| 768 | 10 | exact_maha | Delta_high ~= Delta_low (|D| <= 0.01 and the 95% CI of D inside [-0.02, 0.02]) | -0.0345 | [-0.0363, -0.0327] | fail |
| 2560 | 1 | exact_maha | Delta_high ~= Delta_low (|D| <= 0.01 and the 95% CI of D inside [-0.02, 0.02]) | -0.0298 | [-0.0323, -0.0274] | fail |
| 2560 | 10 | exact_maha | Delta_high ~= Delta_low (|D| <= 0.01 and the 95% CI of D inside [-0.02, 0.02]) | -0.0751 | [-0.0791, -0.0710] | fail |
| 768 | 1 | knn | Delta_high > Delta_low (D > 0) | +0.1858 | [+0.1742, +0.1975] | hold |
| 768 | 10 | knn | Delta_high > Delta_low (D > 0) | +0.1615 | [+0.1528, +0.1701] | hold |
| 2560 | 1 | knn | Delta_high > Delta_low (D > 0) | +0.1499 | [+0.1423, +0.1576] | hold |
| 2560 | 10 | knn | Delta_high > Delta_low (D > 0) | +0.1417 | [+0.1352, +0.1482] | hold |
| 768 | 1 | lw_maha | in between: D_exact < D_lw < D_knn (point estimates) | +0.0062 | [+0.0029, +0.0095] | hold |
| 768 | 10 | lw_maha | in between: D_exact < D_lw < D_knn (point estimates) | -0.0328 | [-0.0346, -0.0311] | hold |
| 2560 | 1 | lw_maha | in between: D_exact < D_lw < D_knn (point estimates) | -0.0385 | [-0.0405, -0.0364] | fail |
| 2560 | 10 | lw_maha | in between: D_exact < D_lw < D_knn (point estimates) | -0.0746 | [-0.0786, -0.0705] | hold |
| 768 | 1 | vim | Delta_low > Delta_high (D < 0) | -0.0146 | [-0.0177, -0.0116] | hold |
| 768 | 10 | vim | Delta_low > Delta_high (D < 0) | -0.0984 | [-0.1032, -0.0935] | hold |
| 2560 | 1 | vim | Delta_low > Delta_high (D < 0) | -0.0423 | [-0.0446, -0.0400] | hold |
| 2560 | 10 | vim | Delta_low > Delta_high (D < 0) | -0.0883 | [-0.0929, -0.0837] | hold |

Verdict: knn 4/4 hold; exact_maha 1/4 hold; lw_maha 3/4 hold; vim 4/4 hold (per PRECOMMIT decision rule).

Caveats: post-hoc (not in the original precommit); synthetic Gaussian features.
