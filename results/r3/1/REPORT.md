# R3 item 1: paper CIs on real features (paper_2fold, Track A scorers)

Commit: 65721c0

Verdict: vs 0.02 (seen = strict): pass_both 208, pass_old_only 35, pass_new_only 0, fail_both 73 of 316 rows; not for the manuscript until item 2 passes.

## Stop-check: package point estimate vs Track A Δ_fit (tolerance 0.001)

| dataset | scorer | cells | match: Track A scorers, strict seen set | max abs diff: Track A scorers, strict seen set | match: package scorers, strict seen set | max abs diff: package scorers, strict seen set |
|---|---|---|---|---|---|---|
| breakhis | knn | 85 | 85/85 | 9.3e-07 | 0/85 | 8.2e-02 |
| breakhis | mahalanobis | 85 | 85/85 | 8.4e-07 | 17/85 | 8.9e-02 |
| breakhis | vim | 85 | 55/85 | 1.3e-01 | 2/85 | 1.5e-01 |
| camelyon | knn | 27 | 27/27 | 9.1e-08 | 1/4 | 5.9e-03 |
| camelyon | mahalanobis | 27 | 27/27 | 8.2e-09 | 0/4 | 9.4e-02 |
| camelyon | vim | 27 | 19/27 | 6.5e-02 | 0/4 | 9.6e-02 |
| dermamnist | knn | 17 | 17/17 | 3.2e-07 | 0/17 | 1.2e-02 |
| dermamnist | mahalanobis | 17 | 17/17 | 3.2e-07 | 6/17 | 3.3e-02 |
| dermamnist | vim | 17 | 13/17 | 9.9e-03 | 0/17 | 3.0e-02 |
| isic2019 | knn | 17 | 12/17 | 1.3e-03 | 0/17 | 2.2e-02 |
| isic2019 | mahalanobis | 17 | 2/17 | 2.7e-03 | 4/17 | 2.5e-02 |
| isic2019 | vim | 17 | 8/17 | 2.9e-03 | 3/17 | 2.1e-02 |
| kermany | knn | 12 | 12/12 | 1.9e-04 | 0/12 | 3.4e-02 |
| kermany | mahalanobis | 12 | 12/12 | 2.0e-04 | 0/12 | 4.2e-02 |
| kermany | vim | 12 | 9/12 | 7.8e-03 | 0/12 | 2.0e-02 |

Mahalanobis / kNN cells failing 0.001 with Track A scorers and Track A seen set (strict where no fold_id_eval): 20

| cell | scorer | Track A Δ | package Δ | abs diff |
|---|---|---|---|---|
| isic2019_convnext_tiny_s42_std | mahalanobis | 0.04256 | 0.04432 | 1.76e-03 |
| isic2019_convnext_tiny_s43_std | mahalanobis | 0.04406 | 0.04575 | 1.69e-03 |
| isic2019_densenet121_s42_std | mahalanobis | 0.03453 | 0.03580 | 1.27e-03 |
| isic2019_densenet121_s42_std | knn | 0.02342 | 0.02448 | 1.07e-03 |
| isic2019_densenet121_s43_std | mahalanobis | 0.03564 | 0.03702 | 1.37e-03 |
| isic2019_densenet121_s43_std | knn | 0.02465 | 0.02590 | 1.25e-03 |
| isic2019_effb3_s42_std | mahalanobis | 0.04583 | 0.04751 | 1.68e-03 |
| isic2019_efficientnet_v2_s_s42_std | mahalanobis | 0.03332 | 0.03445 | 1.13e-03 |
| isic2019_fm_dinov2_vitl14_s42_std | mahalanobis | 0.02646 | 0.02748 | 1.02e-03 |
| isic2019_fm_uni_s42_std | mahalanobis | 0.03550 | 0.03693 | 1.43e-03 |
| isic2019_fm_virchow2_s42_std | mahalanobis | 0.05077 | 0.05274 | 1.97e-03 |
| isic2019_mobilenet_v3_large_s42_std | mahalanobis | 0.03583 | 0.03727 | 1.44e-03 |
| isic2019_regnet_y_3_2gf_s42_std | mahalanobis | 0.06666 | 0.06938 | 2.71e-03 |
| isic2019_regnet_y_3_2gf_s42_std | knn | 0.02986 | 0.03120 | 1.33e-03 |
| isic2019_resnet18_s42_std | mahalanobis | 0.02705 | 0.02842 | 1.37e-03 |
| isic2019_resnet18_s42_std | knn | 0.01964 | 0.02071 | 1.07e-03 |
| isic2019_resnet18_s43_std | mahalanobis | 0.02998 | 0.03123 | 1.24e-03 |
| isic2019_resnet18_s43_std | knn | 0.02072 | 0.02195 | 1.23e-03 |
| isic2019_resnet50_s42_std | mahalanobis | 0.05749 | 0.05999 | 2.50e-03 |
| isic2019_resnet50_s43_std | mahalanobis | 0.05558 | 0.05780 | 2.22e-03 |

## pass_old_only rows

| cell | scorer | seen | Δ | Track A Δ | old CI (bootstrap) | new CI (jackknife) | status vs 0.02 | status vs 0 | n_groups_fit (per fold) | K | d | ID acc | orphan seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kermany_resnet18_s43_std | knn_mean_cosine | strict | 0.0263 | 0.0262 | [0.0212, 0.0315] | [0.0189, 0.0336] | pass_old_only | pass_both | 4240/2 | 2 | 512 | 0.9985 | 0 |
| dermamnist_resnet18_s42_std | knn_mean_cosine | strict | 0.0065 | 0.0065 | [0.0026, 0.0098] | [-0.0001, 0.0130] | fail_both | pass_old_only | 5678/2 | 2 | 512 | 0.8384 | 0 |
| dermamnist_resnet50_s42_std | knn_mean_cosine | strict | 0.0047 | 0.0047 | [0.0016, 0.0079] | [-0.0009, 0.0103] | fail_both | pass_old_only | 5678/2 | 2 | 2048 | 0.8470 | 0 |
| breakhis_convnext_tiny_s42_std_r0 | knn_mean_cosine | strict | 0.0776 | 0.0776 | [0.0516, 0.1031] | [0.0162, 0.1389] | pass_old_only | pass_both | 56/2 | 2 | 768 | 0.9390 | 0 |
| breakhis_convnext_tiny_s42_std_r3 | knn_mean_cosine | strict | 0.0782 | 0.0782 | [0.0468, 0.1125] | [0.0187, 0.1378] | pass_old_only | pass_both | 58/2 | 2 | 768 | 0.9253 | 0 |
| breakhis_convnext_tiny_s43_std_r0 | knn_mean_cosine | strict | 0.0724 | 0.0724 | [0.0460, 0.1021] | [0.0152, 0.1295] | pass_old_only | pass_both | 56/2 | 2 | 768 | 0.9361 | 0 |
| breakhis_effb3_s42_std_r0 | knn_mean_cosine | strict | 0.1025 | 0.1025 | [0.0717, 0.1355] | [0.0160, 0.1889] | pass_old_only | pass_both | 56/2 | 2 | 1536 | 0.9528 | 0 |
| breakhis_effb3_s42_std_r3 | knn_mean_cosine | strict | 0.0964 | 0.0964 | [0.0622, 0.1349] | [0.0101, 0.1827] | pass_old_only | pass_both | 58/2 | 2 | 1536 | 0.9366 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r0 | knn_mean_cosine | strict | 0.0520 | 0.0520 | [0.0329, 0.0730] | [-0.0256, 0.1295] | pass_old_only | pass_old_only | 56/2 | 2 | 1280 | 0.9263 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r1 | knn_mean_cosine | strict | 0.0992 | 0.0992 | [0.0676, 0.1381] | [0.0129, 0.1856] | pass_old_only | pass_both | 56/2 | 2 | 1280 | 0.8946 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r2 | knn_mean_cosine | strict | 0.0864 | 0.0864 | [0.0559, 0.1226] | [-0.0101, 0.1828] | pass_old_only | pass_old_only | 57/2 | 2 | 1280 | 0.8887 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r3 | mahalanobis_l2 | strict | 0.0959 | 0.0959 | [0.0593, 0.1445] | [0.0144, 0.1774] | pass_old_only | pass_both | 58/2 | 2 | 1280 | 0.9244 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r3 | knn_mean_cosine | strict | 0.0520 | 0.0520 | [0.0269, 0.0853] | [-0.0353, 0.1392] | pass_old_only | pass_old_only | 58/2 | 2 | 1280 | 0.9244 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r4 | knn_mean_cosine | strict | 0.0733 | 0.0733 | [0.0489, 0.1021] | [0.0156, 0.1310] | pass_old_only | pass_both | 61/2 | 2 | 1280 | 0.9076 | 0 |
| dermamnist_fm_conch_v1_5_s42_std | knn_mean_cosine | strict | 0.0031 | 0.0031 | [0.0011, 0.0054] | [-0.0010, 0.0073] | fail_both | pass_old_only | 5678/2 | 2 | 768 | 0.7316 | 0 |
| dermamnist_fm_virchow2_s42_std | knn_mean_cosine | strict | 0.0023 | 0.0023 | [0.0009, 0.0037] | [-0.0000, 0.0045] | fail_both | pass_old_only | 5678/2 | 2 | 2560 | 0.7388 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r4 | knn_mean_cosine | strict | 0.0343 | 0.0343 | [0.0235, 0.0462] | [0.0186, 0.0500] | pass_old_only | pass_both | 61/2 | 2 | 768 | 0.7966 | 0 |
| kermany_densenet121_s43_std | mahalanobis_l2 | tracka | 0.0246 | 0.0246 | [0.0204, 0.0290] | [0.0191, 0.0301] | pass_old_only | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 2 |
| isic2019_densenet121_s42_std | knn_mean_cosine | tracka | 0.0234 | 0.0234 | [0.0207, 0.0264] | [0.0198, 0.0270] | pass_old_only | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 117 |
| camelyon_convnext_tiny_s42 | mahalanobis_l2 | strict | 0.0593 | 0.0593 | [0.0178, 0.1038] | [-0.0199, 0.1385] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9969 | 0 |
| camelyon_convnext_tiny_s42 | knn_mean_cosine | strict | 0.0541 | 0.0541 | [0.0308, 0.0802] | [0.0075, 0.1007] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9969 | 0 |
| camelyon_densenet121_s42 | mahalanobis_l2 | strict | 0.1130 | 0.1130 | [0.0284, 0.1540] | [0.0004, 0.2257] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9964 | 0 |
| camelyon_effb3_s42 | mahalanobis_l2 | strict | 0.1402 | 0.1402 | [0.0453, 0.1863] | [0.0160, 0.2643] | pass_old_only | pass_both | 30/2 | 2 | 1536 | 0.9963 | 0 |
| camelyon_efficientnet_v2_s_s42 | mahalanobis_l2 | strict | 0.0831 | 0.0831 | [0.0218, 0.1177] | [0.0123, 0.1539] | pass_old_only | pass_both | 30/2 | 2 | 1280 | 0.9963 | 0 |
| camelyon_mobilenet_v3_large_s42 | mahalanobis_l2 | strict | 0.0988 | 0.0988 | [0.0385, 0.1428] | [0.0180, 0.1796] | pass_old_only | pass_both | 30/2 | 2 | 960 | 0.9965 | 0 |
| camelyon_regnet_y_3_2gf_s42 | mahalanobis_l2 | strict | 0.1492 | 0.1492 | [0.0481, 0.2107] | [0.0057, 0.2927] | pass_old_only | pass_both | 30/2 | 2 | 1512 | 0.9965 | 0 |
| camelyon_resnet18_s42 | mahalanobis_l2 | strict | 0.1683 | 0.1683 | [0.0368, 0.2418] | [-0.0051, 0.3416] | pass_old_only | pass_old_only | 30/2 | 2 | 512 | 0.9954 | 0 |
| camelyon_resnet18_s42 | knn_mean_cosine | strict | 0.1358 | 0.1358 | [0.0437, 0.2071] | [0.0082, 0.2634] | pass_old_only | pass_both | 30/2 | 2 | 512 | 0.9954 | 0 |
| camelyon_resnet50_s43 | mahalanobis_l2 | strict | 0.0776 | 0.0776 | [0.0134, 0.1157] | [-0.0042, 0.1595] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9952 | 0 |
| camelyon_resnet50_s43 | knn_mean_cosine | strict | 0.1319 | 0.1319 | [0.0166, 0.2136] | [-0.0515, 0.3152] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9952 | 0 |
| camelyon_convnext_tiny_s43 | mahalanobis_l2 | strict | 0.0632 | 0.0632 | [0.0177, 0.0959] | [-0.0044, 0.1308] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_resnet50_s44 | mahalanobis_l2 | strict | 0.0697 | 0.0697 | [0.0082, 0.1398] | [-0.0721, 0.2116] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9954 | 0 |
| camelyon_resnet50_s44 | knn_mean_cosine | strict | 0.0902 | 0.0902 | [0.0170, 0.1726] | [-0.0514, 0.2317] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9954 | 0 |
| camelyon_convnext_tiny_s44 | mahalanobis_l2 | strict | 0.0599 | 0.0599 | [0.0181, 0.0988] | [-0.0030, 0.1229] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_convnext_tiny_s44 | knn_mean_cosine | strict | 0.0773 | 0.0773 | [0.0445, 0.1139] | [0.0162, 0.1385] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_uni_s42 | mahalanobis_l2 | strict | 0.0077 | 0.0077 | [0.0021, 0.0132] | [-0.0091, 0.0245] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s42 | knn_mean_cosine | strict | 0.0933 | 0.0933 | [0.0144, 0.1776] | [-0.0207, 0.2073] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s43 | mahalanobis_l2 | strict | 0.0034 | 0.0034 | [0.0007, 0.0062] | [-0.0015, 0.0083] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s43 | knn_mean_cosine | strict | 0.1133 | 0.1133 | [0.0130, 0.1809] | [-0.0410, 0.2676] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s44 | mahalanobis_l2 | strict | 0.0045 | 0.0045 | [0.0010, 0.0085] | [-0.0023, 0.0113] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s44 | knn_mean_cosine | strict | 0.1050 | 0.1050 | [0.0079, 0.2132] | [-0.0490, 0.2591] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_virchow2_s43 | knn_mean_cosine | strict | 0.1525 | 0.1525 | [0.0334, 0.2324] | [-0.0195, 0.3245] | pass_old_only | pass_old_only | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s44 | knn_mean_cosine | strict | 0.1168 | 0.1168 | [0.0264, 0.2133] | [-0.0194, 0.2530] | pass_old_only | pass_old_only | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_dinov2_vitb14_s43 | mahalanobis_l2 | strict | 0.0912 | 0.0912 | [0.0316, 0.1243] | [0.0074, 0.1750] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s44 | mahalanobis_l2 | strict | 0.0585 | 0.0585 | [0.0280, 0.0866] | [0.0126, 0.1043] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s44 | knn_mean_cosine | strict | 0.1016 | 0.1016 | [0.0448, 0.1533] | [0.0183, 0.1850] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitl14_s43 | mahalanobis_l2 | strict | 0.0891 | 0.0891 | [0.0365, 0.1207] | [0.0142, 0.1639] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s43 | knn_mean_cosine | strict | 0.1258 | 0.1258 | [0.0527, 0.1677] | [0.0170, 0.2347] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s44 | mahalanobis_l2 | strict | 0.0631 | 0.0631 | [0.0292, 0.0967] | [0.0046, 0.1216] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s44 | knn_mean_cosine | strict | 0.0947 | 0.0947 | [0.0434, 0.1432] | [0.0136, 0.1758] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_conch_v1_5_s42 | mahalanobis_l2 | strict | 0.0970 | 0.0970 | [0.0333, 0.1576] | [0.0081, 0.1859] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s43 | mahalanobis_l2 | strict | 0.1050 | 0.1050 | [0.0255, 0.1594] | [-0.0078, 0.2178] | pass_old_only | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s43 | knn_mean_cosine | strict | 0.1907 | 0.1907 | [0.0347, 0.2934] | [-0.0278, 0.4092] | pass_old_only | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s44 | mahalanobis_l2 | strict | 0.0808 | 0.0808 | [0.0187, 0.1468] | [-0.0131, 0.1747] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s44 | knn_mean_cosine | strict | 0.1451 | 0.1451 | [0.0257, 0.2733] | [-0.0338, 0.3239] | pass_old_only | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |

## Priority cells (UNI, Virchow2, DermaMNIST, Kermany, ISIC 2019)

| cell | scorer | seen | Δ | Track A Δ | old CI (bootstrap) | new CI (jackknife) | status vs 0.02 | status vs 0 | n_groups_fit (per fold) | K | d | ID acc | orphan seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kermany_convnext_tiny_s42_std | mahalanobis_l2 | strict | 0.0278 | 0.0277 | [0.0217, 0.0344] | [0.0214, 0.0342] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s42_std | knn_mean_cosine | strict | 0.0359 | 0.0358 | [0.0276, 0.0443] | [0.0275, 0.0442] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s43_std | mahalanobis_l2 | strict | 0.0291 | 0.0291 | [0.0232, 0.0353] | [0.0232, 0.0350] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s43_std | knn_mean_cosine | strict | 0.0434 | 0.0433 | [0.0355, 0.0511] | [0.0356, 0.0511] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_densenet121_s42_std | mahalanobis_l2 | strict | 0.0287 | 0.0286 | [0.0231, 0.0344] | [0.0225, 0.0349] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 0 |
| kermany_densenet121_s42_std | knn_mean_cosine | strict | 0.0317 | 0.0316 | [0.0247, 0.0385] | [0.0242, 0.0391] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 0 |
| kermany_densenet121_s43_std | mahalanobis_l2 | strict | 0.0246 | 0.0246 | [0.0205, 0.0290] | [0.0201, 0.0291] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 0 |
| kermany_densenet121_s43_std | knn_mean_cosine | strict | 0.0297 | 0.0296 | [0.0242, 0.0350] | [0.0228, 0.0366] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 0 |
| kermany_effb3_s42_std | mahalanobis_l2 | strict | 0.0205 | 0.0204 | [0.0172, 0.0235] | [0.0171, 0.0238] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 0 |
| kermany_effb3_s42_std | knn_mean_cosine | strict | 0.0211 | 0.0210 | [0.0168, 0.0256] | [0.0165, 0.0256] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 0 |
| kermany_efficientnet_v2_s_s42_std | mahalanobis_l2 | strict | 0.0232 | 0.0232 | [0.0189, 0.0275] | [0.0191, 0.0273] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 0 |
| kermany_efficientnet_v2_s_s42_std | knn_mean_cosine | strict | 0.0187 | 0.0186 | [0.0149, 0.0229] | [0.0145, 0.0229] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 0 |
| kermany_mobilenet_v3_large_s42_std | mahalanobis_l2 | strict | 0.0208 | 0.0208 | [0.0171, 0.0245] | [0.0175, 0.0241] | fail_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 0 |
| kermany_mobilenet_v3_large_s42_std | knn_mean_cosine | strict | 0.0293 | 0.0293 | [0.0234, 0.0355] | [0.0234, 0.0351] | pass_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 0 |
| kermany_regnet_y_3_2gf_s42_std | mahalanobis_l2 | strict | 0.0532 | 0.0530 | [0.0452, 0.0613] | [0.0435, 0.0628] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 0 |
| kermany_regnet_y_3_2gf_s42_std | knn_mean_cosine | strict | 0.0628 | 0.0626 | [0.0501, 0.0759] | [0.0488, 0.0769] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 0 |
| kermany_resnet18_s42_std | mahalanobis_l2 | strict | 0.0166 | 0.0165 | [0.0131, 0.0199] | [0.0124, 0.0208] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 0 |
| kermany_resnet18_s42_std | knn_mean_cosine | strict | 0.0195 | 0.0195 | [0.0151, 0.0238] | [0.0139, 0.0252] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 0 |
| kermany_resnet18_s43_std | mahalanobis_l2 | strict | 0.0243 | 0.0242 | [0.0196, 0.0288] | [0.0193, 0.0293] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9985 | 0 |
| kermany_resnet18_s43_std | knn_mean_cosine | strict | 0.0263 | 0.0262 | [0.0212, 0.0315] | [0.0189, 0.0336] | pass_old_only | pass_both | 4240/2 | 2 | 512 | 0.9985 | 0 |
| kermany_resnet50_s42_std | mahalanobis_l2 | strict | 0.0446 | 0.0445 | [0.0385, 0.0510] | [0.0377, 0.0516] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 0 |
| kermany_resnet50_s42_std | knn_mean_cosine | strict | 0.0205 | 0.0205 | [0.0154, 0.0253] | [0.0138, 0.0271] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 0 |
| kermany_resnet50_s43_std | mahalanobis_l2 | strict | 0.0416 | 0.0415 | [0.0360, 0.0473] | [0.0367, 0.0465] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 0 |
| kermany_resnet50_s43_std | knn_mean_cosine | strict | 0.0182 | 0.0182 | [0.0146, 0.0221] | [0.0138, 0.0227] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 0 |
| isic2019_convnext_tiny_s42_std | mahalanobis_l2 | strict | 0.0443 | 0.0426 | [0.0406, 0.0480] | [0.0404, 0.0482] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 0 |
| isic2019_convnext_tiny_s42_std | knn_mean_cosine | strict | 0.0204 | 0.0194 | [0.0183, 0.0224] | [0.0179, 0.0228] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 0 |
| isic2019_convnext_tiny_s43_std | mahalanobis_l2 | strict | 0.0458 | 0.0441 | [0.0421, 0.0495] | [0.0418, 0.0497] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 0 |
| isic2019_convnext_tiny_s43_std | knn_mean_cosine | strict | 0.0196 | 0.0188 | [0.0176, 0.0215] | [0.0171, 0.0221] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 0 |
| isic2019_densenet121_s42_std | mahalanobis_l2 | strict | 0.0358 | 0.0345 | [0.0324, 0.0394] | [0.0318, 0.0398] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 0 |
| isic2019_densenet121_s42_std | knn_mean_cosine | strict | 0.0245 | 0.0234 | [0.0214, 0.0273] | [0.0211, 0.0279] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 0 |
| isic2019_densenet121_s43_std | mahalanobis_l2 | strict | 0.0370 | 0.0356 | [0.0335, 0.0403] | [0.0333, 0.0407] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 0 |
| isic2019_densenet121_s43_std | knn_mean_cosine | strict | 0.0259 | 0.0247 | [0.0227, 0.0290] | [0.0215, 0.0303] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 0 |
| isic2019_effb3_s42_std | mahalanobis_l2 | strict | 0.0475 | 0.0458 | [0.0438, 0.0516] | [0.0434, 0.0516] | pass_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 0 |
| isic2019_effb3_s42_std | knn_mean_cosine | strict | 0.0179 | 0.0172 | [0.0159, 0.0199] | [0.0151, 0.0207] | fail_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 0 |
| isic2019_efficientnet_v2_s_s42_std | mahalanobis_l2 | strict | 0.0344 | 0.0333 | [0.0316, 0.0374] | [0.0314, 0.0375] | pass_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 0 |
| isic2019_efficientnet_v2_s_s42_std | knn_mean_cosine | strict | 0.0092 | 0.0089 | [0.0079, 0.0107] | [0.0075, 0.0109] | fail_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 0 |
| isic2019_mobilenet_v3_large_s42_std | mahalanobis_l2 | strict | 0.0373 | 0.0358 | [0.0343, 0.0403] | [0.0339, 0.0407] | pass_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 0 |
| isic2019_mobilenet_v3_large_s42_std | knn_mean_cosine | strict | 0.0212 | 0.0203 | [0.0186, 0.0237] | [0.0179, 0.0245] | fail_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 0 |
| isic2019_regnet_y_3_2gf_s42_std | mahalanobis_l2 | strict | 0.0694 | 0.0667 | [0.0640, 0.0750] | [0.0635, 0.0752] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 0 |
| isic2019_regnet_y_3_2gf_s42_std | knn_mean_cosine | strict | 0.0312 | 0.0299 | [0.0278, 0.0344] | [0.0270, 0.0354] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 0 |
| isic2019_resnet18_s42_std | mahalanobis_l2 | strict | 0.0284 | 0.0270 | [0.0258, 0.0311] | [0.0247, 0.0322] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 0 |
| isic2019_resnet18_s42_std | knn_mean_cosine | strict | 0.0207 | 0.0196 | [0.0182, 0.0234] | [0.0170, 0.0245] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 0 |
| isic2019_resnet18_s43_std | mahalanobis_l2 | strict | 0.0312 | 0.0300 | [0.0285, 0.0343] | [0.0270, 0.0354] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 0 |
| isic2019_resnet18_s43_std | knn_mean_cosine | strict | 0.0219 | 0.0207 | [0.0195, 0.0245] | [0.0185, 0.0254] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 0 |
| isic2019_resnet50_s42_std | mahalanobis_l2 | strict | 0.0600 | 0.0575 | [0.0546, 0.0658] | [0.0540, 0.0659] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 0 |
| isic2019_resnet50_s42_std | knn_mean_cosine | strict | 0.0154 | 0.0146 | [0.0129, 0.0179] | [0.0123, 0.0185] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 0 |
| isic2019_resnet50_s43_std | mahalanobis_l2 | strict | 0.0578 | 0.0556 | [0.0527, 0.0631] | [0.0518, 0.0638] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 0 |
| isic2019_resnet50_s43_std | knn_mean_cosine | strict | 0.0177 | 0.0167 | [0.0153, 0.0201] | [0.0140, 0.0213] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 0 |
| dermamnist_convnext_tiny_s42_std | mahalanobis_l2 | strict | 0.0196 | 0.0196 | [0.0160, 0.0232] | [0.0150, 0.0242] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.9235 | 0 |
| dermamnist_convnext_tiny_s42_std | knn_mean_cosine | strict | 0.0051 | 0.0051 | [0.0037, 0.0066] | [0.0030, 0.0071] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.9235 | 0 |
| dermamnist_convnext_tiny_s43_std | mahalanobis_l2 | strict | 0.0258 | 0.0258 | [0.0217, 0.0301] | [0.0215, 0.0302] | pass_both | pass_both | 5678/2 | 2 | 768 | 0.9177 | 0 |
| dermamnist_convnext_tiny_s43_std | knn_mean_cosine | strict | 0.0057 | 0.0057 | [0.0041, 0.0072] | [0.0035, 0.0078] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.9177 | 0 |
| dermamnist_densenet121_s42_std | mahalanobis_l2 | strict | 0.0158 | 0.0158 | [0.0118, 0.0200] | [0.0097, 0.0219] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8745 | 0 |
| dermamnist_densenet121_s42_std | knn_mean_cosine | strict | 0.0062 | 0.0062 | [0.0031, 0.0092] | [0.0018, 0.0106] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8745 | 0 |
| dermamnist_densenet121_s43_std | mahalanobis_l2 | strict | 0.0138 | 0.0138 | [0.0098, 0.0185] | [0.0085, 0.0190] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8716 | 0 |
| dermamnist_densenet121_s43_std | knn_mean_cosine | strict | 0.0051 | 0.0051 | [0.0025, 0.0080] | [0.0007, 0.0094] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8716 | 0 |
| dermamnist_effb3_s42_std | mahalanobis_l2 | strict | 0.0235 | 0.0235 | [0.0191, 0.0282] | [0.0181, 0.0289] | fail_both | pass_both | 5678/2 | 2 | 1536 | 0.8817 | 0 |
| dermamnist_effb3_s42_std | knn_mean_cosine | strict | 0.0078 | 0.0078 | [0.0055, 0.0100] | [0.0047, 0.0108] | fail_both | pass_both | 5678/2 | 2 | 1536 | 0.8817 | 0 |
| dermamnist_efficientnet_v2_s_s42_std | mahalanobis_l2 | strict | 0.0141 | 0.0141 | [0.0104, 0.0183] | [0.0092, 0.0190] | fail_both | pass_both | 5678/2 | 2 | 1280 | 0.8990 | 0 |
| dermamnist_efficientnet_v2_s_s42_std | knn_mean_cosine | strict | 0.0041 | 0.0041 | [0.0028, 0.0054] | [0.0021, 0.0061] | fail_both | pass_both | 5678/2 | 2 | 1280 | 0.8990 | 0 |
| dermamnist_mobilenet_v3_large_s42_std | mahalanobis_l2 | strict | 0.0281 | 0.0281 | [0.0227, 0.0332] | [0.0222, 0.0340] | pass_both | pass_both | 5678/2 | 2 | 960 | 0.8268 | 0 |
| dermamnist_mobilenet_v3_large_s42_std | knn_mean_cosine | strict | 0.0127 | 0.0127 | [0.0085, 0.0166] | [0.0071, 0.0184] | fail_both | pass_both | 5678/2 | 2 | 960 | 0.8268 | 0 |
| dermamnist_regnet_y_3_2gf_s42_std | mahalanobis_l2 | strict | 0.0292 | 0.0292 | [0.0237, 0.0354] | [0.0213, 0.0372] | pass_both | pass_both | 5678/2 | 2 | 1512 | 0.8672 | 0 |
| dermamnist_regnet_y_3_2gf_s42_std | knn_mean_cosine | strict | 0.0114 | 0.0114 | [0.0080, 0.0150] | [0.0063, 0.0166] | fail_both | pass_both | 5678/2 | 2 | 1512 | 0.8672 | 0 |
| dermamnist_resnet18_s42_std | mahalanobis_l2 | strict | 0.0214 | 0.0214 | [0.0165, 0.0265] | [0.0150, 0.0279] | fail_both | pass_both | 5678/2 | 2 | 512 | 0.8384 | 0 |
| dermamnist_resnet18_s42_std | knn_mean_cosine | strict | 0.0065 | 0.0065 | [0.0026, 0.0098] | [-0.0001, 0.0130] | fail_both | pass_old_only | 5678/2 | 2 | 512 | 0.8384 | 0 |
| dermamnist_resnet18_s43_std | mahalanobis_l2 | strict | 0.0285 | 0.0285 | [0.0230, 0.0343] | [0.0206, 0.0364] | pass_both | pass_both | 5678/2 | 2 | 512 | 0.8398 | 0 |
| dermamnist_resnet18_s43_std | knn_mean_cosine | strict | 0.0100 | 0.0100 | [0.0062, 0.0141] | [0.0044, 0.0156] | fail_both | pass_both | 5678/2 | 2 | 512 | 0.8398 | 0 |
| dermamnist_resnet50_s42_std | mahalanobis_l2 | strict | 0.0206 | 0.0206 | [0.0168, 0.0244] | [0.0152, 0.0259] | fail_both | pass_both | 5678/2 | 2 | 2048 | 0.8470 | 0 |
| dermamnist_resnet50_s42_std | knn_mean_cosine | strict | 0.0047 | 0.0047 | [0.0016, 0.0079] | [-0.0009, 0.0103] | fail_both | pass_old_only | 5678/2 | 2 | 2048 | 0.8470 | 0 |
| dermamnist_resnet50_s43_std | mahalanobis_l2 | strict | 0.0181 | 0.0181 | [0.0142, 0.0224] | [0.0127, 0.0235] | fail_both | pass_both | 5678/2 | 2 | 2048 | 0.8716 | 0 |
| dermamnist_resnet50_s43_std | knn_mean_cosine | strict | 0.0063 | 0.0063 | [0.0042, 0.0084] | [0.0028, 0.0098] | fail_both | pass_both | 5678/2 | 2 | 2048 | 0.8716 | 0 |
| dermamnist_fm_conch_v1_5_s42_std | mahalanobis_l2 | strict | 0.0079 | 0.0079 | [0.0054, 0.0105] | [0.0041, 0.0117] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.7316 | 0 |
| dermamnist_fm_conch_v1_5_s42_std | knn_mean_cosine | strict | 0.0031 | 0.0031 | [0.0011, 0.0054] | [-0.0010, 0.0073] | fail_both | pass_old_only | 5678/2 | 2 | 768 | 0.7316 | 0 |
| dermamnist_fm_dinov2_vitb14_s42_std | mahalanobis_l2 | strict | 0.0126 | 0.0126 | [0.0099, 0.0156] | [0.0094, 0.0158] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.7172 | 0 |
| dermamnist_fm_dinov2_vitb14_s42_std | knn_mean_cosine | strict | 0.0078 | 0.0078 | [0.0059, 0.0098] | [0.0046, 0.0110] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.7172 | 0 |
| dermamnist_fm_dinov2_vitl14_s42_std | mahalanobis_l2 | strict | 0.0142 | 0.0142 | [0.0107, 0.0182] | [0.0103, 0.0181] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.6782 | 0 |
| dermamnist_fm_dinov2_vitl14_s42_std | knn_mean_cosine | strict | 0.0067 | 0.0067 | [0.0048, 0.0087] | [0.0040, 0.0094] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.6782 | 0 |
| dermamnist_fm_uni_s42_std | mahalanobis_l2 | strict | 0.0084 | 0.0084 | [0.0054, 0.0115] | [0.0050, 0.0117] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.7201 | 0 |
| dermamnist_fm_uni_s42_std | knn_mean_cosine | strict | 0.0028 | 0.0028 | [0.0017, 0.0041] | [0.0010, 0.0046] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.7201 | 0 |
| dermamnist_fm_virchow2_s42_std | mahalanobis_l2 | strict | 0.0120 | 0.0120 | [0.0087, 0.0154] | [0.0079, 0.0161] | fail_both | pass_both | 5678/2 | 2 | 2560 | 0.7388 | 0 |
| dermamnist_fm_virchow2_s42_std | knn_mean_cosine | strict | 0.0023 | 0.0023 | [0.0009, 0.0037] | [-0.0000, 0.0045] | fail_both | pass_old_only | 5678/2 | 2 | 2560 | 0.7388 | 0 |
| isic2019_fm_conch_v1_5_s42_std | mahalanobis_l2 | strict | 0.0176 | 0.0172 | [0.0160, 0.0193] | [0.0159, 0.0194] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 0 |
| isic2019_fm_conch_v1_5_s42_std | knn_mean_cosine | strict | 0.0170 | 0.0162 | [0.0149, 0.0191] | [0.0141, 0.0198] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 0 |
| isic2019_fm_dinov2_vitb14_s42_std | mahalanobis_l2 | strict | 0.0209 | 0.0202 | [0.0193, 0.0226] | [0.0190, 0.0228] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 0 |
| isic2019_fm_dinov2_vitb14_s42_std | knn_mean_cosine | strict | 0.0131 | 0.0126 | [0.0116, 0.0147] | [0.0110, 0.0152] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 0 |
| isic2019_fm_dinov2_vitl14_s42_std | mahalanobis_l2 | strict | 0.0275 | 0.0265 | [0.0255, 0.0294] | [0.0249, 0.0301] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 0 |
| isic2019_fm_dinov2_vitl14_s42_std | knn_mean_cosine | strict | 0.0134 | 0.0129 | [0.0117, 0.0149] | [0.0115, 0.0153] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 0 |
| isic2019_fm_uni_s42_std | mahalanobis_l2 | strict | 0.0369 | 0.0355 | [0.0342, 0.0397] | [0.0341, 0.0398] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 0 |
| isic2019_fm_uni_s42_std | knn_mean_cosine | strict | 0.0182 | 0.0175 | [0.0161, 0.0201] | [0.0158, 0.0205] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 0 |
| isic2019_fm_virchow2_s42_std | mahalanobis_l2 | strict | 0.0527 | 0.0508 | [0.0490, 0.0563] | [0.0489, 0.0566] | pass_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 0 |
| isic2019_fm_virchow2_s42_std | knn_mean_cosine | strict | 0.0128 | 0.0126 | [0.0110, 0.0146] | [0.0105, 0.0151] | fail_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 0 |
| breakhis_fm_uni_s42_std_r0 | mahalanobis_l2 | strict | 0.2191 | 0.2191 | [0.1899, 0.2514] | [0.1715, 0.2667] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9607 | 0 |
| breakhis_fm_uni_s42_std_r0 | knn_mean_cosine | strict | 0.1508 | 0.1508 | [0.1244, 0.1758] | [0.1101, 0.1915] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9607 | 0 |
| breakhis_fm_uni_s42_std_r1 | mahalanobis_l2 | strict | 0.2863 | 0.2863 | [0.2407, 0.3389] | [0.2069, 0.3657] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9122 | 0 |
| breakhis_fm_uni_s42_std_r1 | knn_mean_cosine | strict | 0.2184 | 0.2184 | [0.1806, 0.2615] | [0.1474, 0.2894] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9122 | 0 |
| breakhis_fm_uni_s42_std_r2 | mahalanobis_l2 | strict | 0.2455 | 0.2455 | [0.2047, 0.2919] | [0.1803, 0.3107] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.9190 | 0 |
| breakhis_fm_uni_s42_std_r2 | knn_mean_cosine | strict | 0.1843 | 0.1843 | [0.1492, 0.2194] | [0.1318, 0.2368] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.9190 | 0 |
| breakhis_fm_uni_s42_std_r3 | mahalanobis_l2 | strict | 0.2279 | 0.2279 | [0.1900, 0.2670] | [0.1746, 0.2813] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9374 | 0 |
| breakhis_fm_uni_s42_std_r3 | knn_mean_cosine | strict | 0.1506 | 0.1506 | [0.1225, 0.1780] | [0.1028, 0.1984] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9374 | 0 |
| breakhis_fm_uni_s42_std_r4 | mahalanobis_l2 | strict | 0.2438 | 0.2438 | [0.2056, 0.2866] | [0.1916, 0.2961] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.9178 | 0 |
| breakhis_fm_uni_s42_std_r4 | knn_mean_cosine | strict | 0.1677 | 0.1677 | [0.1367, 0.1994] | [0.1196, 0.2158] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.9178 | 0 |
| breakhis_fm_virchow2_s42_std_r0 | mahalanobis_l2 | strict | 0.2026 | 0.2026 | [0.1681, 0.2410] | [0.1418, 0.2635] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9813 | 0 |
| breakhis_fm_virchow2_s42_std_r0 | knn_mean_cosine | strict | 0.1094 | 0.1094 | [0.0844, 0.1348] | [0.0688, 0.1500] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9813 | 0 |
| breakhis_fm_virchow2_s42_std_r1 | mahalanobis_l2 | strict | 0.2501 | 0.2501 | [0.2052, 0.2981] | [0.1659, 0.3342] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9353 | 0 |
| breakhis_fm_virchow2_s42_std_r1 | knn_mean_cosine | strict | 0.1424 | 0.1424 | [0.1112, 0.1744] | [0.0788, 0.2060] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9353 | 0 |
| breakhis_fm_virchow2_s42_std_r2 | mahalanobis_l2 | strict | 0.2309 | 0.2309 | [0.1880, 0.2753] | [0.1550, 0.3069] | pass_both | pass_both | 57/2 | 2 | 2560 | 0.9338 | 0 |
| breakhis_fm_virchow2_s42_std_r2 | knn_mean_cosine | strict | 0.1304 | 0.1304 | [0.1020, 0.1591] | [0.0775, 0.1833] | pass_both | pass_both | 57/2 | 2 | 2560 | 0.9338 | 0 |
| breakhis_fm_virchow2_s42_std_r3 | mahalanobis_l2 | strict | 0.2024 | 0.2024 | [0.1709, 0.2347] | [0.1495, 0.2554] | pass_both | pass_both | 58/2 | 2 | 2560 | 0.9461 | 0 |
| breakhis_fm_virchow2_s42_std_r3 | knn_mean_cosine | strict | 0.1025 | 0.1025 | [0.0816, 0.1229] | [0.0622, 0.1428] | pass_both | pass_both | 58/2 | 2 | 2560 | 0.9461 | 0 |
| breakhis_fm_virchow2_s42_std_r4 | mahalanobis_l2 | strict | 0.2128 | 0.2128 | [0.1790, 0.2508] | [0.1543, 0.2712] | pass_both | pass_both | 61/2 | 2 | 2560 | 0.9364 | 0 |
| breakhis_fm_virchow2_s42_std_r4 | knn_mean_cosine | strict | 0.1084 | 0.1084 | [0.0876, 0.1306] | [0.0732, 0.1436] | pass_both | pass_both | 61/2 | 2 | 2560 | 0.9364 | 0 |
| kermany_convnext_tiny_s42_std | mahalanobis_l2 | tracka | 0.0277 | 0.0277 | [0.0218, 0.0341] | [0.0219, 0.0336] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_convnext_tiny_s42_std | knn_mean_cosine | tracka | 0.0358 | 0.0358 | [0.0280, 0.0442] | [0.0270, 0.0446] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_convnext_tiny_s43_std | mahalanobis_l2 | tracka | 0.0291 | 0.0291 | [0.0230, 0.0354] | [0.0232, 0.0349] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_convnext_tiny_s43_std | knn_mean_cosine | tracka | 0.0433 | 0.0433 | [0.0353, 0.0504] | [0.0356, 0.0509] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_densenet121_s42_std | mahalanobis_l2 | tracka | 0.0286 | 0.0286 | [0.0230, 0.0340] | [0.0218, 0.0355] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 2 |
| kermany_densenet121_s42_std | knn_mean_cosine | tracka | 0.0316 | 0.0316 | [0.0245, 0.0383] | [0.0237, 0.0395] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 2 |
| kermany_densenet121_s43_std | mahalanobis_l2 | tracka | 0.0246 | 0.0246 | [0.0204, 0.0290] | [0.0191, 0.0301] | pass_old_only | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 2 |
| kermany_densenet121_s43_std | knn_mean_cosine | tracka | 0.0296 | 0.0296 | [0.0243, 0.0351] | [0.0221, 0.0372] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 2 |
| kermany_effb3_s42_std | mahalanobis_l2 | tracka | 0.0204 | 0.0204 | [0.0172, 0.0237] | [0.0172, 0.0236] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 2 |
| kermany_effb3_s42_std | knn_mean_cosine | tracka | 0.0210 | 0.0210 | [0.0168, 0.0256] | [0.0164, 0.0257] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 2 |
| kermany_efficientnet_v2_s_s42_std | mahalanobis_l2 | tracka | 0.0232 | 0.0232 | [0.0189, 0.0277] | [0.0183, 0.0280] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 2 |
| kermany_efficientnet_v2_s_s42_std | knn_mean_cosine | tracka | 0.0186 | 0.0186 | [0.0148, 0.0228] | [0.0137, 0.0236] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 2 |
| kermany_mobilenet_v3_large_s42_std | mahalanobis_l2 | tracka | 0.0208 | 0.0208 | [0.0172, 0.0247] | [0.0167, 0.0248] | fail_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 2 |
| kermany_mobilenet_v3_large_s42_std | knn_mean_cosine | tracka | 0.0293 | 0.0293 | [0.0234, 0.0353] | [0.0226, 0.0359] | pass_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 2 |
| kermany_regnet_y_3_2gf_s42_std | mahalanobis_l2 | tracka | 0.0530 | 0.0530 | [0.0447, 0.0612] | [0.0440, 0.0619] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 2 |
| kermany_regnet_y_3_2gf_s42_std | knn_mean_cosine | tracka | 0.0626 | 0.0626 | [0.0500, 0.0746] | [0.0469, 0.0784] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 2 |
| kermany_resnet18_s42_std | mahalanobis_l2 | tracka | 0.0165 | 0.0165 | [0.0130, 0.0197] | [0.0120, 0.0211] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 2 |
| kermany_resnet18_s42_std | knn_mean_cosine | tracka | 0.0195 | 0.0195 | [0.0150, 0.0235] | [0.0138, 0.0251] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 2 |
| kermany_resnet18_s43_std | mahalanobis_l2 | tracka | 0.0242 | 0.0242 | [0.0198, 0.0287] | [0.0189, 0.0296] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9985 | 2 |
| kermany_resnet18_s43_std | knn_mean_cosine | tracka | 0.0262 | 0.0262 | [0.0208, 0.0319] | [0.0200, 0.0324] | pass_both | pass_both | 4240/2 | 2 | 512 | 0.9985 | 2 |
| kermany_resnet50_s42_std | mahalanobis_l2 | tracka | 0.0445 | 0.0445 | [0.0376, 0.0504] | [0.0374, 0.0516] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 2 |
| kermany_resnet50_s42_std | knn_mean_cosine | tracka | 0.0205 | 0.0205 | [0.0156, 0.0248] | [0.0139, 0.0271] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 2 |
| kermany_resnet50_s43_std | mahalanobis_l2 | tracka | 0.0415 | 0.0415 | [0.0358, 0.0468] | [0.0359, 0.0471] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 2 |
| kermany_resnet50_s43_std | knn_mean_cosine | tracka | 0.0182 | 0.0182 | [0.0147, 0.0221] | [0.0129, 0.0235] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 2 |
| isic2019_convnext_tiny_s42_std | mahalanobis_l2 | tracka | 0.0426 | 0.0426 | [0.0391, 0.0462] | [0.0386, 0.0465] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 117 |
| isic2019_convnext_tiny_s42_std | knn_mean_cosine | tracka | 0.0194 | 0.0194 | [0.0174, 0.0214] | [0.0169, 0.0219] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 117 |
| isic2019_convnext_tiny_s43_std | mahalanobis_l2 | tracka | 0.0441 | 0.0441 | [0.0405, 0.0478] | [0.0398, 0.0483] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 117 |
| isic2019_convnext_tiny_s43_std | knn_mean_cosine | tracka | 0.0188 | 0.0188 | [0.0167, 0.0207] | [0.0161, 0.0214] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 117 |
| isic2019_densenet121_s42_std | mahalanobis_l2 | tracka | 0.0345 | 0.0345 | [0.0315, 0.0379] | [0.0305, 0.0386] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 117 |
| isic2019_densenet121_s42_std | knn_mean_cosine | tracka | 0.0234 | 0.0234 | [0.0207, 0.0264] | [0.0198, 0.0270] | pass_old_only | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 117 |
| isic2019_densenet121_s43_std | mahalanobis_l2 | tracka | 0.0356 | 0.0356 | [0.0324, 0.0389] | [0.0318, 0.0394] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 117 |
| isic2019_densenet121_s43_std | knn_mean_cosine | tracka | 0.0247 | 0.0247 | [0.0218, 0.0276] | [0.0209, 0.0284] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 117 |
| isic2019_effb3_s42_std | mahalanobis_l2 | tracka | 0.0458 | 0.0458 | [0.0423, 0.0494] | [0.0419, 0.0498] | pass_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 117 |
| isic2019_effb3_s42_std | knn_mean_cosine | tracka | 0.0172 | 0.0172 | [0.0153, 0.0192] | [0.0145, 0.0199] | fail_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 117 |
| isic2019_efficientnet_v2_s_s42_std | mahalanobis_l2 | tracka | 0.0333 | 0.0333 | [0.0307, 0.0361] | [0.0302, 0.0364] | pass_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 117 |
| isic2019_efficientnet_v2_s_s42_std | knn_mean_cosine | tracka | 0.0089 | 0.0089 | [0.0076, 0.0102] | [0.0073, 0.0105] | fail_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 117 |
| isic2019_mobilenet_v3_large_s42_std | mahalanobis_l2 | tracka | 0.0358 | 0.0358 | [0.0330, 0.0386] | [0.0322, 0.0394] | pass_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 117 |
| isic2019_mobilenet_v3_large_s42_std | knn_mean_cosine | tracka | 0.0203 | 0.0203 | [0.0181, 0.0228] | [0.0168, 0.0238] | fail_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 117 |
| isic2019_regnet_y_3_2gf_s42_std | mahalanobis_l2 | tracka | 0.0667 | 0.0667 | [0.0616, 0.0719] | [0.0608, 0.0725] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 117 |
| isic2019_regnet_y_3_2gf_s42_std | knn_mean_cosine | tracka | 0.0299 | 0.0299 | [0.0268, 0.0329] | [0.0262, 0.0336] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 117 |
| isic2019_resnet18_s42_std | mahalanobis_l2 | tracka | 0.0270 | 0.0270 | [0.0245, 0.0297] | [0.0236, 0.0305] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 117 |
| isic2019_resnet18_s42_std | knn_mean_cosine | tracka | 0.0196 | 0.0196 | [0.0172, 0.0221] | [0.0161, 0.0232] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 117 |
| isic2019_resnet18_s43_std | mahalanobis_l2 | tracka | 0.0300 | 0.0300 | [0.0270, 0.0327] | [0.0265, 0.0335] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 117 |
| isic2019_resnet18_s43_std | knn_mean_cosine | tracka | 0.0207 | 0.0207 | [0.0183, 0.0232] | [0.0175, 0.0239] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 117 |
| isic2019_resnet50_s42_std | mahalanobis_l2 | tracka | 0.0575 | 0.0575 | [0.0529, 0.0628] | [0.0523, 0.0627] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 117 |
| isic2019_resnet50_s42_std | knn_mean_cosine | tracka | 0.0146 | 0.0146 | [0.0123, 0.0169] | [0.0111, 0.0180] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 117 |
| isic2019_resnet50_s43_std | mahalanobis_l2 | tracka | 0.0556 | 0.0556 | [0.0509, 0.0604] | [0.0502, 0.0610] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 117 |
| isic2019_resnet50_s43_std | knn_mean_cosine | tracka | 0.0167 | 0.0167 | [0.0145, 0.0192] | [0.0140, 0.0195] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 117 |
| isic2019_fm_conch_v1_5_s42_std | mahalanobis_l2 | tracka | 0.0172 | 0.0172 | [0.0156, 0.0188] | [0.0153, 0.0191] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 117 |
| isic2019_fm_conch_v1_5_s42_std | knn_mean_cosine | tracka | 0.0162 | 0.0162 | [0.0141, 0.0184] | [0.0131, 0.0192] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 117 |
| isic2019_fm_dinov2_vitb14_s42_std | mahalanobis_l2 | tracka | 0.0202 | 0.0202 | [0.0186, 0.0219] | [0.0182, 0.0223] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 117 |
| isic2019_fm_dinov2_vitb14_s42_std | knn_mean_cosine | tracka | 0.0126 | 0.0126 | [0.0110, 0.0143] | [0.0105, 0.0147] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 117 |
| isic2019_fm_dinov2_vitl14_s42_std | mahalanobis_l2 | tracka | 0.0265 | 0.0265 | [0.0245, 0.0284] | [0.0241, 0.0288] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 117 |
| isic2019_fm_dinov2_vitl14_s42_std | knn_mean_cosine | tracka | 0.0129 | 0.0129 | [0.0113, 0.0146] | [0.0109, 0.0149] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 117 |
| isic2019_fm_uni_s42_std | mahalanobis_l2 | tracka | 0.0355 | 0.0355 | [0.0330, 0.0380] | [0.0323, 0.0387] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 117 |
| isic2019_fm_uni_s42_std | knn_mean_cosine | tracka | 0.0175 | 0.0175 | [0.0156, 0.0193] | [0.0146, 0.0204] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 117 |
| isic2019_fm_virchow2_s42_std | mahalanobis_l2 | tracka | 0.0508 | 0.0508 | [0.0473, 0.0543] | [0.0466, 0.0549] | pass_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 117 |
| isic2019_fm_virchow2_s42_std | knn_mean_cosine | tracka | 0.0126 | 0.0126 | [0.0107, 0.0142] | [0.0095, 0.0157] | fail_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 117 |
| camelyon_uni_s42 | mahalanobis_l2 | strict | 0.0077 | 0.0077 | [0.0021, 0.0132] | [-0.0091, 0.0245] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s42 | knn_mean_cosine | strict | 0.0933 | 0.0933 | [0.0144, 0.1776] | [-0.0207, 0.2073] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s43 | mahalanobis_l2 | strict | 0.0034 | 0.0034 | [0.0007, 0.0062] | [-0.0015, 0.0083] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s43 | knn_mean_cosine | strict | 0.1133 | 0.1133 | [0.0130, 0.1809] | [-0.0410, 0.2676] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s44 | mahalanobis_l2 | strict | 0.0045 | 0.0045 | [0.0010, 0.0085] | [-0.0023, 0.0113] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s44 | knn_mean_cosine | strict | 0.1050 | 0.1050 | [0.0079, 0.2132] | [-0.0490, 0.2591] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_virchow2_s42 | mahalanobis_l2 | strict | 0.0325 | 0.0325 | [0.0141, 0.0524] | [0.0083, 0.0568] | fail_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s42 | knn_mean_cosine | strict | 0.1121 | 0.1121 | [0.0377, 0.1914] | [0.0248, 0.1993] | pass_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s43 | mahalanobis_l2 | strict | 0.0255 | 0.0255 | [0.0086, 0.0375] | [0.0041, 0.0470] | fail_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s43 | knn_mean_cosine | strict | 0.1525 | 0.1525 | [0.0334, 0.2324] | [-0.0195, 0.3245] | pass_old_only | pass_old_only | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s44 | mahalanobis_l2 | strict | 0.0226 | 0.0226 | [0.0067, 0.0387] | [0.0038, 0.0414] | fail_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s44 | knn_mean_cosine | strict | 0.1168 | 0.1168 | [0.0264, 0.2133] | [-0.0194, 0.2530] | pass_old_only | pass_old_only | 30/2 | 2 | 2560 | 0.9948 | 0 |

## All rows

| cell | scorer | seen | Δ | Track A Δ | old CI (bootstrap) | new CI (jackknife) | status vs 0.02 | status vs 0 | n_groups_fit (per fold) | K | d | ID acc | orphan seen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| breakhis_convnext_tiny_s42_std_r0 | knn_mean_cosine | strict | 0.0776 | 0.0776 | [0.0516, 0.1031] | [0.0162, 0.1389] | pass_old_only | pass_both | 56/2 | 2 | 768 | 0.9390 | 0 |
| breakhis_convnext_tiny_s42_std_r0 | mahalanobis_l2 | strict | 0.1218 | 0.1218 | [0.0886, 0.1562] | [0.0633, 0.1803] | pass_both | pass_both | 56/2 | 2 | 768 | 0.9390 | 0 |
| breakhis_convnext_tiny_s42_std_r1 | knn_mean_cosine | strict | 0.1192 | 0.1192 | [0.0796, 0.1647] | [0.0433, 0.1951] | pass_both | pass_both | 56/2 | 2 | 768 | 0.9030 | 0 |
| breakhis_convnext_tiny_s42_std_r1 | mahalanobis_l2 | strict | 0.1560 | 0.1560 | [0.1200, 0.1960] | [0.0913, 0.2208] | pass_both | pass_both | 56/2 | 2 | 768 | 0.9030 | 0 |
| breakhis_convnext_tiny_s42_std_r2 | knn_mean_cosine | strict | 0.0877 | 0.0877 | [0.0588, 0.1175] | [0.0227, 0.1528] | pass_both | pass_both | 57/2 | 2 | 768 | 0.9080 | 0 |
| breakhis_convnext_tiny_s42_std_r2 | mahalanobis_l2 | strict | 0.1307 | 0.1307 | [0.0961, 0.1677] | [0.0703, 0.1911] | pass_both | pass_both | 57/2 | 2 | 768 | 0.9080 | 0 |
| breakhis_convnext_tiny_s42_std_r3 | knn_mean_cosine | strict | 0.0782 | 0.0782 | [0.0468, 0.1125] | [0.0187, 0.1378] | pass_old_only | pass_both | 58/2 | 2 | 768 | 0.9253 | 0 |
| breakhis_convnext_tiny_s42_std_r3 | mahalanobis_l2 | strict | 0.1080 | 0.1080 | [0.0798, 0.1396] | [0.0584, 0.1576] | pass_both | pass_both | 58/2 | 2 | 768 | 0.9253 | 0 |
| breakhis_convnext_tiny_s42_std_r4 | knn_mean_cosine | strict | 0.0856 | 0.0856 | [0.0619, 0.1134] | [0.0360, 0.1352] | pass_both | pass_both | 61/2 | 2 | 768 | 0.9068 | 0 |
| breakhis_convnext_tiny_s42_std_r4 | mahalanobis_l2 | strict | 0.1274 | 0.1274 | [0.0992, 0.1591] | [0.0713, 0.1835] | pass_both | pass_both | 61/2 | 2 | 768 | 0.9068 | 0 |
| breakhis_convnext_tiny_s43_std_r0 | knn_mean_cosine | strict | 0.0724 | 0.0724 | [0.0460, 0.1021] | [0.0152, 0.1295] | pass_old_only | pass_both | 56/2 | 2 | 768 | 0.9361 | 0 |
| breakhis_convnext_tiny_s43_std_r0 | mahalanobis_l2 | strict | 0.1205 | 0.1205 | [0.0832, 0.1639] | [0.0657, 0.1753] | pass_both | pass_both | 56/2 | 2 | 768 | 0.9361 | 0 |
| breakhis_convnext_tiny_s43_std_r1 | knn_mean_cosine | strict | 0.1183 | 0.1183 | [0.0765, 0.1633] | [0.0477, 0.1889] | pass_both | pass_both | 56/2 | 2 | 768 | 0.9020 | 0 |
| breakhis_convnext_tiny_s43_std_r1 | mahalanobis_l2 | strict | 0.1673 | 0.1673 | [0.1216, 0.2164] | [0.0989, 0.2357] | pass_both | pass_both | 56/2 | 2 | 768 | 0.9020 | 0 |
| breakhis_convnext_tiny_s43_std_r2 | knn_mean_cosine | strict | 0.0987 | 0.0987 | [0.0630, 0.1413] | [0.0200, 0.1775] | pass_both | pass_both | 57/2 | 2 | 768 | 0.9043 | 0 |
| breakhis_convnext_tiny_s43_std_r2 | mahalanobis_l2 | strict | 0.1352 | 0.1352 | [0.1009, 0.1714] | [0.0680, 0.2025] | pass_both | pass_both | 57/2 | 2 | 768 | 0.9043 | 0 |
| breakhis_convnext_tiny_s43_std_r3 | knn_mean_cosine | strict | 0.0874 | 0.0874 | [0.0571, 0.1230] | [0.0320, 0.1428] | pass_both | pass_both | 58/2 | 2 | 768 | 0.9114 | 0 |
| breakhis_convnext_tiny_s43_std_r3 | mahalanobis_l2 | strict | 0.1094 | 0.1094 | [0.0831, 0.1413] | [0.0626, 0.1562] | pass_both | pass_both | 58/2 | 2 | 768 | 0.9114 | 0 |
| breakhis_convnext_tiny_s43_std_r4 | knn_mean_cosine | strict | 0.0941 | 0.0941 | [0.0669, 0.1229] | [0.0502, 0.1380] | pass_both | pass_both | 61/2 | 2 | 768 | 0.9144 | 0 |
| breakhis_convnext_tiny_s43_std_r4 | mahalanobis_l2 | strict | 0.1343 | 0.1343 | [0.0986, 0.1719] | [0.0775, 0.1911] | pass_both | pass_both | 61/2 | 2 | 768 | 0.9144 | 0 |
| breakhis_densenet121_s42_std_r0 | knn_mean_cosine | strict | 0.1559 | 0.1559 | [0.1065, 0.2084] | [0.0669, 0.2448] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9223 | 0 |
| breakhis_densenet121_s42_std_r0 | mahalanobis_l2 | strict | 0.2155 | 0.2155 | [0.1615, 0.2685] | [0.1277, 0.3034] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9223 | 0 |
| breakhis_densenet121_s42_std_r1 | knn_mean_cosine | strict | 0.2447 | 0.2447 | [0.1841, 0.3085] | [0.1300, 0.3593] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.8845 | 0 |
| breakhis_densenet121_s42_std_r1 | mahalanobis_l2 | strict | 0.3241 | 0.3241 | [0.2588, 0.3944] | [0.2090, 0.4391] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.8845 | 0 |
| breakhis_densenet121_s42_std_r2 | knn_mean_cosine | strict | 0.1875 | 0.1875 | [0.1357, 0.2426] | [0.0736, 0.3013] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.8979 | 0 |
| breakhis_densenet121_s42_std_r2 | mahalanobis_l2 | strict | 0.2670 | 0.2670 | [0.2066, 0.3312] | [0.1483, 0.3857] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.8979 | 0 |
| breakhis_densenet121_s42_std_r3 | knn_mean_cosine | strict | 0.1284 | 0.1284 | [0.0846, 0.1777] | [0.0358, 0.2210] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9114 | 0 |
| breakhis_densenet121_s42_std_r3 | mahalanobis_l2 | strict | 0.2029 | 0.2029 | [0.1523, 0.2595] | [0.1131, 0.2927] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9114 | 0 |
| breakhis_densenet121_s42_std_r4 | knn_mean_cosine | strict | 0.1810 | 0.1810 | [0.1311, 0.2354] | [0.0853, 0.2768] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.9000 | 0 |
| breakhis_densenet121_s42_std_r4 | mahalanobis_l2 | strict | 0.2552 | 0.2552 | [0.1969, 0.3201] | [0.1626, 0.3479] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.9000 | 0 |
| breakhis_densenet121_s43_std_r0 | knn_mean_cosine | strict | 0.1591 | 0.1591 | [0.1125, 0.2137] | [0.0812, 0.2371] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9105 | 0 |
| breakhis_densenet121_s43_std_r0 | mahalanobis_l2 | strict | 0.2279 | 0.2279 | [0.1774, 0.2864] | [0.1481, 0.3076] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9105 | 0 |
| breakhis_densenet121_s43_std_r1 | knn_mean_cosine | strict | 0.2278 | 0.2278 | [0.1632, 0.2950] | [0.1190, 0.3367] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9067 | 0 |
| breakhis_densenet121_s43_std_r1 | mahalanobis_l2 | strict | 0.3174 | 0.3174 | [0.2459, 0.3857] | [0.2036, 0.4313] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9067 | 0 |
| breakhis_densenet121_s43_std_r2 | knn_mean_cosine | strict | 0.1870 | 0.1870 | [0.1341, 0.2445] | [0.0661, 0.3078] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.8979 | 0 |
| breakhis_densenet121_s43_std_r2 | mahalanobis_l2 | strict | 0.2641 | 0.2641 | [0.2004, 0.3318] | [0.1361, 0.3921] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.8979 | 0 |
| breakhis_densenet121_s43_std_r3 | knn_mean_cosine | strict | 0.1254 | 0.1254 | [0.0835, 0.1744] | [0.0410, 0.2098] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9157 | 0 |
| breakhis_densenet121_s43_std_r3 | mahalanobis_l2 | strict | 0.1962 | 0.1962 | [0.1443, 0.2529] | [0.1063, 0.2861] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9157 | 0 |
| breakhis_densenet121_s43_std_r4 | knn_mean_cosine | strict | 0.1824 | 0.1824 | [0.1264, 0.2375] | [0.0960, 0.2688] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.8975 | 0 |
| breakhis_densenet121_s43_std_r4 | mahalanobis_l2 | strict | 0.2448 | 0.2448 | [0.1846, 0.3065] | [0.1674, 0.3222] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.8975 | 0 |
| breakhis_effb3_s42_std_r0 | knn_mean_cosine | strict | 0.1025 | 0.1025 | [0.0717, 0.1355] | [0.0160, 0.1889] | pass_old_only | pass_both | 56/2 | 2 | 1536 | 0.9528 | 0 |
| breakhis_effb3_s42_std_r0 | mahalanobis_l2 | strict | 0.1430 | 0.1430 | [0.1092, 0.1807] | [0.0714, 0.2145] | pass_both | pass_both | 56/2 | 2 | 1536 | 0.9528 | 0 |
| breakhis_effb3_s42_std_r1 | knn_mean_cosine | strict | 0.1571 | 0.1571 | [0.1190, 0.2023] | [0.0702, 0.2441] | pass_both | pass_both | 56/2 | 2 | 1536 | 0.9085 | 0 |
| breakhis_effb3_s42_std_r1 | mahalanobis_l2 | strict | 0.2310 | 0.2310 | [0.1817, 0.2898] | [0.1486, 0.3135] | pass_both | pass_both | 56/2 | 2 | 1536 | 0.9085 | 0 |
| breakhis_effb3_s42_std_r2 | knn_mean_cosine | strict | 0.1332 | 0.1332 | [0.0939, 0.1764] | [0.0431, 0.2234] | pass_both | pass_both | 57/2 | 2 | 1536 | 0.9135 | 0 |
| breakhis_effb3_s42_std_r2 | mahalanobis_l2 | strict | 0.1890 | 0.1890 | [0.1337, 0.2481] | [0.0910, 0.2870] | pass_both | pass_both | 57/2 | 2 | 1536 | 0.9135 | 0 |
| breakhis_effb3_s42_std_r3 | knn_mean_cosine | strict | 0.0964 | 0.0964 | [0.0622, 0.1349] | [0.0101, 0.1827] | pass_old_only | pass_both | 58/2 | 2 | 1536 | 0.9366 | 0 |
| breakhis_effb3_s42_std_r3 | mahalanobis_l2 | strict | 0.1412 | 0.1412 | [0.0982, 0.1873] | [0.0630, 0.2193] | pass_both | pass_both | 58/2 | 2 | 1536 | 0.9366 | 0 |
| breakhis_effb3_s42_std_r4 | knn_mean_cosine | strict | 0.1167 | 0.1167 | [0.0868, 0.1552] | [0.0551, 0.1783] | pass_both | pass_both | 61/2 | 2 | 1536 | 0.9144 | 0 |
| breakhis_effb3_s42_std_r4 | mahalanobis_l2 | strict | 0.1713 | 0.1713 | [0.1293, 0.2188] | [0.0999, 0.2427] | pass_both | pass_both | 61/2 | 2 | 1536 | 0.9144 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r0 | knn_mean_cosine | strict | 0.0520 | 0.0520 | [0.0329, 0.0730] | [-0.0256, 0.1295] | pass_old_only | pass_old_only | 56/2 | 2 | 1280 | 0.9263 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r0 | mahalanobis_l2 | strict | 0.1009 | 0.1009 | [0.0701, 0.1327] | [0.0340, 0.1677] | pass_both | pass_both | 56/2 | 2 | 1280 | 0.9263 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r1 | knn_mean_cosine | strict | 0.0992 | 0.0992 | [0.0676, 0.1381] | [0.0129, 0.1856] | pass_old_only | pass_both | 56/2 | 2 | 1280 | 0.8946 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r1 | mahalanobis_l2 | strict | 0.1587 | 0.1587 | [0.1109, 0.2175] | [0.0704, 0.2469] | pass_both | pass_both | 56/2 | 2 | 1280 | 0.8946 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r2 | knn_mean_cosine | strict | 0.0864 | 0.0864 | [0.0559, 0.1226] | [-0.0101, 0.1828] | pass_old_only | pass_old_only | 57/2 | 2 | 1280 | 0.8887 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r2 | mahalanobis_l2 | strict | 0.1398 | 0.1398 | [0.0971, 0.1907] | [0.0382, 0.2415] | pass_both | pass_both | 57/2 | 2 | 1280 | 0.8887 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r3 | knn_mean_cosine | strict | 0.0520 | 0.0520 | [0.0269, 0.0853] | [-0.0353, 0.1392] | pass_old_only | pass_old_only | 58/2 | 2 | 1280 | 0.9244 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r3 | mahalanobis_l2 | strict | 0.0959 | 0.0959 | [0.0593, 0.1445] | [0.0144, 0.1774] | pass_old_only | pass_both | 58/2 | 2 | 1280 | 0.9244 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r4 | knn_mean_cosine | strict | 0.0733 | 0.0733 | [0.0489, 0.1021] | [0.0156, 0.1310] | pass_old_only | pass_both | 61/2 | 2 | 1280 | 0.9076 | 0 |
| breakhis_efficientnet_v2_s_s42_std_r4 | mahalanobis_l2 | strict | 0.1237 | 0.1237 | [0.0825, 0.1725] | [0.0537, 0.1936] | pass_both | pass_both | 61/2 | 2 | 1280 | 0.9076 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r0 | knn_mean_cosine | strict | 0.0747 | 0.0747 | [0.0564, 0.0936] | [0.0468, 0.1026] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8977 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r0 | mahalanobis_l2 | strict | 0.1186 | 0.1186 | [0.0992, 0.1371] | [0.0873, 0.1500] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8977 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r1 | knn_mean_cosine | strict | 0.0970 | 0.0970 | [0.0722, 0.1238] | [0.0517, 0.1422] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8567 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r1 | mahalanobis_l2 | strict | 0.1411 | 0.1411 | [0.1156, 0.1679] | [0.0956, 0.1867] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8567 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r2 | knn_mean_cosine | strict | 0.0908 | 0.0908 | [0.0713, 0.1117] | [0.0555, 0.1261] | pass_both | pass_both | 57/2 | 2 | 768 | 0.8648 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r2 | mahalanobis_l2 | strict | 0.1358 | 0.1358 | [0.1106, 0.1631] | [0.0904, 0.1811] | pass_both | pass_both | 57/2 | 2 | 768 | 0.8648 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r3 | knn_mean_cosine | strict | 0.0595 | 0.0595 | [0.0469, 0.0721] | [0.0348, 0.0843] | pass_both | pass_both | 58/2 | 2 | 768 | 0.8853 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r3 | mahalanobis_l2 | strict | 0.1036 | 0.1036 | [0.0856, 0.1222] | [0.0741, 0.1332] | pass_both | pass_both | 58/2 | 2 | 768 | 0.8853 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r4 | knn_mean_cosine | strict | 0.0684 | 0.0684 | [0.0524, 0.0854] | [0.0423, 0.0944] | pass_both | pass_both | 61/2 | 2 | 768 | 0.8737 | 0 |
| breakhis_fm_conch_v1_5_s42_std_r4 | mahalanobis_l2 | strict | 0.1097 | 0.1097 | [0.0891, 0.1310] | [0.0782, 0.1411] | pass_both | pass_both | 61/2 | 2 | 768 | 0.8737 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r0 | knn_mean_cosine | strict | 0.0332 | 0.0332 | [0.0258, 0.0413] | [0.0230, 0.0433] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8496 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r0 | mahalanobis_l2 | strict | 0.0594 | 0.0594 | [0.0497, 0.0699] | [0.0471, 0.0716] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8496 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r1 | knn_mean_cosine | strict | 0.0479 | 0.0479 | [0.0331, 0.0654] | [0.0208, 0.0750] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8226 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r1 | mahalanobis_l2 | strict | 0.0783 | 0.0783 | [0.0593, 0.1005] | [0.0485, 0.1080] | pass_both | pass_both | 56/2 | 2 | 768 | 0.8226 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r2 | knn_mean_cosine | strict | 0.0405 | 0.0405 | [0.0307, 0.0509] | [0.0227, 0.0583] | pass_both | pass_both | 57/2 | 2 | 768 | 0.8123 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r2 | mahalanobis_l2 | strict | 0.0693 | 0.0693 | [0.0525, 0.0903] | [0.0423, 0.0964] | pass_both | pass_both | 57/2 | 2 | 768 | 0.8123 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r3 | knn_mean_cosine | strict | 0.0348 | 0.0348 | [0.0251, 0.0470] | [0.0200, 0.0497] | pass_both | pass_both | 58/2 | 2 | 768 | 0.8480 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r3 | mahalanobis_l2 | strict | 0.0600 | 0.0600 | [0.0452, 0.0793] | [0.0375, 0.0824] | pass_both | pass_both | 58/2 | 2 | 768 | 0.8480 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r4 | knn_mean_cosine | strict | 0.0343 | 0.0343 | [0.0235, 0.0462] | [0.0186, 0.0500] | pass_old_only | pass_both | 61/2 | 2 | 768 | 0.7966 | 0 |
| breakhis_fm_dinov2_vitb14_s42_std_r4 | mahalanobis_l2 | strict | 0.0561 | 0.0561 | [0.0411, 0.0740] | [0.0351, 0.0772] | pass_both | pass_both | 61/2 | 2 | 768 | 0.7966 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r0 | knn_mean_cosine | strict | 0.0411 | 0.0411 | [0.0316, 0.0524] | [0.0273, 0.0550] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.8948 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r0 | mahalanobis_l2 | strict | 0.0723 | 0.0723 | [0.0616, 0.0852] | [0.0564, 0.0883] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.8948 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r1 | knn_mean_cosine | strict | 0.0542 | 0.0542 | [0.0388, 0.0707] | [0.0255, 0.0828] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.8577 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r1 | mahalanobis_l2 | strict | 0.1016 | 0.1016 | [0.0790, 0.1297] | [0.0628, 0.1404] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.8577 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r2 | knn_mean_cosine | strict | 0.0464 | 0.0464 | [0.0351, 0.0591] | [0.0277, 0.0652] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.8574 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r2 | mahalanobis_l2 | strict | 0.0882 | 0.0882 | [0.0675, 0.1162] | [0.0540, 0.1223] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.8574 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r3 | knn_mean_cosine | strict | 0.0401 | 0.0401 | [0.0280, 0.0545] | [0.0222, 0.0579] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.8836 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r3 | mahalanobis_l2 | strict | 0.0783 | 0.0783 | [0.0586, 0.1049] | [0.0485, 0.1082] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.8836 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r4 | knn_mean_cosine | strict | 0.0381 | 0.0381 | [0.0278, 0.0500] | [0.0236, 0.0526] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.8534 | 0 |
| breakhis_fm_dinov2_vitl14_s42_std_r4 | mahalanobis_l2 | strict | 0.0754 | 0.0754 | [0.0562, 0.1003] | [0.0473, 0.1034] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.8534 | 0 |
| breakhis_fm_uni_s42_std_r0 | knn_mean_cosine | strict | 0.1508 | 0.1508 | [0.1244, 0.1758] | [0.1101, 0.1915] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9607 | 0 |
| breakhis_fm_uni_s42_std_r0 | mahalanobis_l2 | strict | 0.2191 | 0.2191 | [0.1899, 0.2514] | [0.1715, 0.2667] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9607 | 0 |
| breakhis_fm_uni_s42_std_r1 | knn_mean_cosine | strict | 0.2184 | 0.2184 | [0.1806, 0.2615] | [0.1474, 0.2894] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9122 | 0 |
| breakhis_fm_uni_s42_std_r1 | mahalanobis_l2 | strict | 0.2863 | 0.2863 | [0.2407, 0.3389] | [0.2069, 0.3657] | pass_both | pass_both | 56/2 | 2 | 1024 | 0.9122 | 0 |
| breakhis_fm_uni_s42_std_r2 | knn_mean_cosine | strict | 0.1843 | 0.1843 | [0.1492, 0.2194] | [0.1318, 0.2368] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.9190 | 0 |
| breakhis_fm_uni_s42_std_r2 | mahalanobis_l2 | strict | 0.2455 | 0.2455 | [0.2047, 0.2919] | [0.1803, 0.3107] | pass_both | pass_both | 57/2 | 2 | 1024 | 0.9190 | 0 |
| breakhis_fm_uni_s42_std_r3 | knn_mean_cosine | strict | 0.1506 | 0.1506 | [0.1225, 0.1780] | [0.1028, 0.1984] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9374 | 0 |
| breakhis_fm_uni_s42_std_r3 | mahalanobis_l2 | strict | 0.2279 | 0.2279 | [0.1900, 0.2670] | [0.1746, 0.2813] | pass_both | pass_both | 58/2 | 2 | 1024 | 0.9374 | 0 |
| breakhis_fm_uni_s42_std_r4 | knn_mean_cosine | strict | 0.1677 | 0.1677 | [0.1367, 0.1994] | [0.1196, 0.2158] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.9178 | 0 |
| breakhis_fm_uni_s42_std_r4 | mahalanobis_l2 | strict | 0.2438 | 0.2438 | [0.2056, 0.2866] | [0.1916, 0.2961] | pass_both | pass_both | 61/2 | 2 | 1024 | 0.9178 | 0 |
| breakhis_fm_virchow2_s42_std_r0 | knn_mean_cosine | strict | 0.1094 | 0.1094 | [0.0844, 0.1348] | [0.0688, 0.1500] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9813 | 0 |
| breakhis_fm_virchow2_s42_std_r0 | mahalanobis_l2 | strict | 0.2026 | 0.2026 | [0.1681, 0.2410] | [0.1418, 0.2635] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9813 | 0 |
| breakhis_fm_virchow2_s42_std_r1 | knn_mean_cosine | strict | 0.1424 | 0.1424 | [0.1112, 0.1744] | [0.0788, 0.2060] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9353 | 0 |
| breakhis_fm_virchow2_s42_std_r1 | mahalanobis_l2 | strict | 0.2501 | 0.2501 | [0.2052, 0.2981] | [0.1659, 0.3342] | pass_both | pass_both | 56/2 | 2 | 2560 | 0.9353 | 0 |
| breakhis_fm_virchow2_s42_std_r2 | knn_mean_cosine | strict | 0.1304 | 0.1304 | [0.1020, 0.1591] | [0.0775, 0.1833] | pass_both | pass_both | 57/2 | 2 | 2560 | 0.9338 | 0 |
| breakhis_fm_virchow2_s42_std_r2 | mahalanobis_l2 | strict | 0.2309 | 0.2309 | [0.1880, 0.2753] | [0.1550, 0.3069] | pass_both | pass_both | 57/2 | 2 | 2560 | 0.9338 | 0 |
| breakhis_fm_virchow2_s42_std_r3 | knn_mean_cosine | strict | 0.1025 | 0.1025 | [0.0816, 0.1229] | [0.0622, 0.1428] | pass_both | pass_both | 58/2 | 2 | 2560 | 0.9461 | 0 |
| breakhis_fm_virchow2_s42_std_r3 | mahalanobis_l2 | strict | 0.2024 | 0.2024 | [0.1709, 0.2347] | [0.1495, 0.2554] | pass_both | pass_both | 58/2 | 2 | 2560 | 0.9461 | 0 |
| breakhis_fm_virchow2_s42_std_r4 | knn_mean_cosine | strict | 0.1084 | 0.1084 | [0.0876, 0.1306] | [0.0732, 0.1436] | pass_both | pass_both | 61/2 | 2 | 2560 | 0.9364 | 0 |
| breakhis_fm_virchow2_s42_std_r4 | mahalanobis_l2 | strict | 0.2128 | 0.2128 | [0.1790, 0.2508] | [0.1543, 0.2712] | pass_both | pass_both | 61/2 | 2 | 2560 | 0.9364 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r0 | knn_mean_cosine | strict | 0.1153 | 0.1153 | [0.0836, 0.1479] | [0.0343, 0.1964] | pass_both | pass_both | 56/2 | 2 | 960 | 0.8968 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r0 | mahalanobis_l2 | strict | 0.1398 | 0.1398 | [0.1040, 0.1737] | [0.0750, 0.2045] | pass_both | pass_both | 56/2 | 2 | 960 | 0.8968 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r1 | knn_mean_cosine | strict | 0.2028 | 0.2028 | [0.1556, 0.2548] | [0.1141, 0.2915] | pass_both | pass_both | 56/2 | 2 | 960 | 0.8734 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r1 | mahalanobis_l2 | strict | 0.2384 | 0.2384 | [0.1874, 0.2962] | [0.1499, 0.3269] | pass_both | pass_both | 56/2 | 2 | 960 | 0.8734 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r2 | knn_mean_cosine | strict | 0.1565 | 0.1565 | [0.1186, 0.1952] | [0.0659, 0.2472] | pass_both | pass_both | 57/2 | 2 | 960 | 0.8648 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r2 | mahalanobis_l2 | strict | 0.1849 | 0.1849 | [0.1425, 0.2347] | [0.0990, 0.2708] | pass_both | pass_both | 57/2 | 2 | 960 | 0.8648 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r3 | knn_mean_cosine | strict | 0.1033 | 0.1033 | [0.0737, 0.1337] | [0.0385, 0.1681] | pass_both | pass_both | 58/2 | 2 | 960 | 0.8940 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r3 | mahalanobis_l2 | strict | 0.1463 | 0.1463 | [0.1065, 0.1941] | [0.0760, 0.2166] | pass_both | pass_both | 58/2 | 2 | 960 | 0.8940 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r4 | knn_mean_cosine | strict | 0.1551 | 0.1551 | [0.1151, 0.2010] | [0.0825, 0.2278] | pass_both | pass_both | 61/2 | 2 | 960 | 0.8873 | 0 |
| breakhis_mobilenet_v3_large_s42_std_r4 | mahalanobis_l2 | strict | 0.1848 | 0.1848 | [0.1407, 0.2338] | [0.1193, 0.2504] | pass_both | pass_both | 61/2 | 2 | 960 | 0.8873 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r0 | knn_mean_cosine | strict | 0.1844 | 0.1844 | [0.1283, 0.2362] | [0.0975, 0.2714] | pass_both | pass_both | 56/2 | 2 | 1512 | 0.9282 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r0 | mahalanobis_l2 | strict | 0.2461 | 0.2461 | [0.1911, 0.2989] | [0.1650, 0.3271] | pass_both | pass_both | 56/2 | 2 | 1512 | 0.9282 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r1 | knn_mean_cosine | strict | 0.2768 | 0.2768 | [0.2135, 0.3453] | [0.1701, 0.3835] | pass_both | pass_both | 56/2 | 2 | 1512 | 0.8900 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r1 | mahalanobis_l2 | strict | 0.3143 | 0.3143 | [0.2460, 0.3887] | [0.2022, 0.4264] | pass_both | pass_both | 56/2 | 2 | 1512 | 0.8900 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r2 | knn_mean_cosine | strict | 0.2157 | 0.2157 | [0.1623, 0.2681] | [0.1097, 0.3218] | pass_both | pass_both | 57/2 | 2 | 1512 | 0.8924 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r2 | mahalanobis_l2 | strict | 0.2541 | 0.2541 | [0.1999, 0.3138] | [0.1399, 0.3683] | pass_both | pass_both | 57/2 | 2 | 1512 | 0.8924 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r3 | knn_mean_cosine | strict | 0.1541 | 0.1541 | [0.1133, 0.1995] | [0.0716, 0.2365] | pass_both | pass_both | 58/2 | 2 | 1512 | 0.9123 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r3 | mahalanobis_l2 | strict | 0.2142 | 0.2142 | [0.1637, 0.2682] | [0.1239, 0.3045] | pass_both | pass_both | 58/2 | 2 | 1512 | 0.9123 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r4 | knn_mean_cosine | strict | 0.2379 | 0.2379 | [0.1748, 0.3055] | [0.1357, 0.3401] | pass_both | pass_both | 61/2 | 2 | 1512 | 0.8958 | 0 |
| breakhis_regnet_y_3_2gf_s42_std_r4 | mahalanobis_l2 | strict | 0.2833 | 0.2833 | [0.2186, 0.3502] | [0.1934, 0.3732] | pass_both | pass_both | 61/2 | 2 | 1512 | 0.8958 | 0 |
| breakhis_resnet18_s42_std_r0 | knn_mean_cosine | strict | 0.1408 | 0.1408 | [0.0927, 0.1922] | [0.0581, 0.2234] | pass_both | pass_both | 56/2 | 2 | 512 | 0.9086 | 0 |
| breakhis_resnet18_s42_std_r0 | mahalanobis_l2 | strict | 0.1845 | 0.1845 | [0.1351, 0.2347] | [0.1089, 0.2601] | pass_both | pass_both | 56/2 | 2 | 512 | 0.9086 | 0 |
| breakhis_resnet18_s42_std_r1 | knn_mean_cosine | strict | 0.2179 | 0.2179 | [0.1592, 0.2793] | [0.1208, 0.3151] | pass_both | pass_both | 56/2 | 2 | 512 | 0.8826 | 0 |
| breakhis_resnet18_s42_std_r1 | mahalanobis_l2 | strict | 0.2726 | 0.2726 | [0.2126, 0.3365] | [0.1750, 0.3702] | pass_both | pass_both | 56/2 | 2 | 512 | 0.8826 | 0 |
| breakhis_resnet18_s42_std_r2 | knn_mean_cosine | strict | 0.1659 | 0.1659 | [0.1258, 0.2066] | [0.0724, 0.2594] | pass_both | pass_both | 57/2 | 2 | 512 | 0.8638 | 0 |
| breakhis_resnet18_s42_std_r2 | mahalanobis_l2 | strict | 0.2075 | 0.2075 | [0.1597, 0.2574] | [0.1161, 0.2989] | pass_both | pass_both | 57/2 | 2 | 512 | 0.8638 | 0 |
| breakhis_resnet18_s42_std_r3 | knn_mean_cosine | strict | 0.1189 | 0.1189 | [0.0819, 0.1575] | [0.0504, 0.1873] | pass_both | pass_both | 58/2 | 2 | 512 | 0.9010 | 0 |
| breakhis_resnet18_s42_std_r3 | mahalanobis_l2 | strict | 0.1613 | 0.1613 | [0.1189, 0.2061] | [0.0948, 0.2277] | pass_both | pass_both | 58/2 | 2 | 512 | 0.9010 | 0 |
| breakhis_resnet18_s42_std_r4 | knn_mean_cosine | strict | 0.1531 | 0.1531 | [0.1112, 0.2029] | [0.0648, 0.2414] | pass_both | pass_both | 61/2 | 2 | 512 | 0.8754 | 0 |
| breakhis_resnet18_s42_std_r4 | mahalanobis_l2 | strict | 0.2011 | 0.2011 | [0.1589, 0.2462] | [0.1263, 0.2758] | pass_both | pass_both | 61/2 | 2 | 512 | 0.8754 | 0 |
| breakhis_resnet18_s43_std_r0 | knn_mean_cosine | strict | 0.1346 | 0.1346 | [0.0903, 0.1889] | [0.0676, 0.2015] | pass_both | pass_both | 56/2 | 2 | 512 | 0.8968 | 0 |
| breakhis_resnet18_s43_std_r0 | mahalanobis_l2 | strict | 0.1868 | 0.1868 | [0.1371, 0.2457] | [0.1166, 0.2571] | pass_both | pass_both | 56/2 | 2 | 512 | 0.8968 | 0 |
| breakhis_resnet18_s43_std_r1 | knn_mean_cosine | strict | 0.2354 | 0.2354 | [0.1731, 0.2967] | [0.1342, 0.3365] | pass_both | pass_both | 56/2 | 2 | 512 | 0.8808 | 0 |
| breakhis_resnet18_s43_std_r1 | mahalanobis_l2 | strict | 0.2826 | 0.2826 | [0.2199, 0.3397] | [0.1825, 0.3827] | pass_both | pass_both | 56/2 | 2 | 512 | 0.8808 | 0 |
| breakhis_resnet18_s43_std_r2 | knn_mean_cosine | strict | 0.1741 | 0.1741 | [0.1307, 0.2269] | [0.0692, 0.2790] | pass_both | pass_both | 57/2 | 2 | 512 | 0.8703 | 0 |
| breakhis_resnet18_s43_std_r2 | mahalanobis_l2 | strict | 0.2178 | 0.2178 | [0.1675, 0.2745] | [0.1179, 0.3176] | pass_both | pass_both | 57/2 | 2 | 512 | 0.8703 | 0 |
| breakhis_resnet18_s43_std_r3 | knn_mean_cosine | strict | 0.1200 | 0.1200 | [0.0805, 0.1642] | [0.0513, 0.1887] | pass_both | pass_both | 58/2 | 2 | 512 | 0.9027 | 0 |
| breakhis_resnet18_s43_std_r3 | mahalanobis_l2 | strict | 0.1639 | 0.1639 | [0.1188, 0.2123] | [0.0959, 0.2319] | pass_both | pass_both | 58/2 | 2 | 512 | 0.9027 | 0 |
| breakhis_resnet18_s43_std_r4 | knn_mean_cosine | strict | 0.1614 | 0.1614 | [0.1145, 0.2098] | [0.0838, 0.2390] | pass_both | pass_both | 61/2 | 2 | 512 | 0.8797 | 0 |
| breakhis_resnet18_s43_std_r4 | mahalanobis_l2 | strict | 0.1997 | 0.1997 | [0.1506, 0.2479] | [0.1378, 0.2616] | pass_both | pass_both | 61/2 | 2 | 512 | 0.8797 | 0 |
| breakhis_resnet50_s42_std_r0 | knn_mean_cosine | strict | 0.1399 | 0.1399 | [0.0943, 0.1865] | [0.0479, 0.2319] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.9213 | 0 |
| breakhis_resnet50_s42_std_r0 | mahalanobis_l2 | strict | 0.2269 | 0.2269 | [0.1732, 0.2814] | [0.1364, 0.3174] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.9213 | 0 |
| breakhis_resnet50_s42_std_r1 | knn_mean_cosine | strict | 0.2215 | 0.2215 | [0.1653, 0.2822] | [0.1040, 0.3390] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.8928 | 0 |
| breakhis_resnet50_s42_std_r1 | mahalanobis_l2 | strict | 0.3237 | 0.3237 | [0.2586, 0.3958] | [0.2025, 0.4449] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.8928 | 0 |
| breakhis_resnet50_s42_std_r2 | knn_mean_cosine | strict | 0.1883 | 0.1883 | [0.1301, 0.2442] | [0.0621, 0.3146] | pass_both | pass_both | 57/2 | 2 | 2048 | 0.8859 | 0 |
| breakhis_resnet50_s42_std_r2 | mahalanobis_l2 | strict | 0.2777 | 0.2777 | [0.2141, 0.3458] | [0.1510, 0.4043] | pass_both | pass_both | 57/2 | 2 | 2048 | 0.8859 | 0 |
| breakhis_resnet50_s42_std_r3 | knn_mean_cosine | strict | 0.1334 | 0.1334 | [0.0895, 0.1809] | [0.0468, 0.2200] | pass_both | pass_both | 58/2 | 2 | 2048 | 0.9096 | 0 |
| breakhis_resnet50_s42_std_r3 | mahalanobis_l2 | strict | 0.2152 | 0.2152 | [0.1678, 0.2652] | [0.1295, 0.3009] | pass_both | pass_both | 58/2 | 2 | 2048 | 0.9096 | 0 |
| breakhis_resnet50_s42_std_r4 | knn_mean_cosine | strict | 0.1840 | 0.1840 | [0.1365, 0.2345] | [0.0927, 0.2752] | pass_both | pass_both | 61/2 | 2 | 2048 | 0.9178 | 0 |
| breakhis_resnet50_s42_std_r4 | mahalanobis_l2 | strict | 0.2784 | 0.2784 | [0.2155, 0.3477] | [0.1749, 0.3819] | pass_both | pass_both | 61/2 | 2 | 2048 | 0.9178 | 0 |
| breakhis_resnet50_s43_std_r0 | knn_mean_cosine | strict | 0.1526 | 0.1526 | [0.1063, 0.2056] | [0.0746, 0.2306] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.9263 | 0 |
| breakhis_resnet50_s43_std_r0 | mahalanobis_l2 | strict | 0.2364 | 0.2364 | [0.1824, 0.3033] | [0.1538, 0.3190] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.9263 | 0 |
| breakhis_resnet50_s43_std_r1 | knn_mean_cosine | strict | 0.2202 | 0.2202 | [0.1582, 0.2842] | [0.1064, 0.3339] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.8928 | 0 |
| breakhis_resnet50_s43_std_r1 | mahalanobis_l2 | strict | 0.3213 | 0.3213 | [0.2482, 0.3950] | [0.1994, 0.4431] | pass_both | pass_both | 56/2 | 2 | 2048 | 0.8928 | 0 |
| breakhis_resnet50_s43_std_r2 | knn_mean_cosine | strict | 0.1945 | 0.1945 | [0.1392, 0.2574] | [0.0646, 0.3244] | pass_both | pass_both | 57/2 | 2 | 2048 | 0.8942 | 0 |
| breakhis_resnet50_s43_std_r2 | mahalanobis_l2 | strict | 0.2788 | 0.2788 | [0.2123, 0.3507] | [0.1496, 0.4080] | pass_both | pass_both | 57/2 | 2 | 2048 | 0.8942 | 0 |
| breakhis_resnet50_s43_std_r3 | knn_mean_cosine | strict | 0.1176 | 0.1176 | [0.0760, 0.1644] | [0.0292, 0.2059] | pass_both | pass_both | 58/2 | 2 | 2048 | 0.9105 | 0 |
| breakhis_resnet50_s43_std_r3 | mahalanobis_l2 | strict | 0.2105 | 0.2105 | [0.1604, 0.2672] | [0.1212, 0.2998] | pass_both | pass_both | 58/2 | 2 | 2048 | 0.9105 | 0 |
| breakhis_resnet50_s43_std_r4 | knn_mean_cosine | strict | 0.1777 | 0.1777 | [0.1280, 0.2276] | [0.0979, 0.2576] | pass_both | pass_both | 61/2 | 2 | 2048 | 0.8949 | 0 |
| breakhis_resnet50_s43_std_r4 | mahalanobis_l2 | strict | 0.2750 | 0.2750 | [0.2094, 0.3444] | [0.1744, 0.3755] | pass_both | pass_both | 61/2 | 2 | 2048 | 0.8949 | 0 |
| camelyon_conch_v1_5_s42 | knn_mean_cosine | strict | 0.1505 | 0.1505 | [0.0438, 0.2523] | [0.0209, 0.2801] | pass_both | pass_both | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s42 | mahalanobis_l2 | strict | 0.0970 | 0.0970 | [0.0333, 0.1576] | [0.0081, 0.1859] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s43 | knn_mean_cosine | strict | 0.1907 | 0.1907 | [0.0347, 0.2934] | [-0.0278, 0.4092] | pass_old_only | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s43 | mahalanobis_l2 | strict | 0.1050 | 0.1050 | [0.0255, 0.1594] | [-0.0078, 0.2178] | pass_old_only | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s44 | knn_mean_cosine | strict | 0.1451 | 0.1451 | [0.0257, 0.2733] | [-0.0338, 0.3239] | pass_old_only | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_conch_v1_5_s44 | mahalanobis_l2 | strict | 0.0808 | 0.0808 | [0.0187, 0.1468] | [-0.0131, 0.1747] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9873 | 0 |
| camelyon_convnext_tiny_s42 | knn_mean_cosine | strict | 0.0541 | 0.0541 | [0.0308, 0.0802] | [0.0075, 0.1007] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9969 | 0 |
| camelyon_convnext_tiny_s42 | mahalanobis_l2 | strict | 0.0593 | 0.0593 | [0.0178, 0.1038] | [-0.0199, 0.1385] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9969 | 0 |
| camelyon_convnext_tiny_s43 | knn_mean_cosine | strict | 0.0708 | 0.0708 | [0.0407, 0.0921] | [0.0216, 0.1200] | pass_both | pass_both | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_convnext_tiny_s43 | mahalanobis_l2 | strict | 0.0632 | 0.0632 | [0.0177, 0.0959] | [-0.0044, 0.1308] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_convnext_tiny_s44 | knn_mean_cosine | strict | 0.0773 | 0.0773 | [0.0445, 0.1139] | [0.0162, 0.1385] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_convnext_tiny_s44 | mahalanobis_l2 | strict | 0.0599 | 0.0599 | [0.0181, 0.0988] | [-0.0030, 0.1229] | fail_both | pass_old_only | 30/2 | 2 | 768 | 0.9966 | 0 |
| camelyon_densenet121_s42 | knn_mean_cosine | strict | 0.0931 | 0.0931 | [0.0298, 0.1253] | [0.0360, 0.1502] | pass_both | pass_both | 30/2 | 2 | 1024 | 0.9964 | 0 |
| camelyon_densenet121_s42 | mahalanobis_l2 | strict | 0.1130 | 0.1130 | [0.0284, 0.1540] | [0.0004, 0.2257] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9964 | 0 |
| camelyon_dinov2_vitb14_s42 | knn_mean_cosine | strict | 0.1123 | 0.1123 | [0.0560, 0.1512] | [0.0639, 0.1607] | pass_both | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s42 | mahalanobis_l2 | strict | 0.0673 | 0.0673 | [0.0369, 0.0853] | [0.0490, 0.0856] | pass_both | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s43 | knn_mean_cosine | strict | 0.1321 | 0.1321 | [0.0545, 0.1774] | [0.0216, 0.2427] | pass_both | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s43 | mahalanobis_l2 | strict | 0.0912 | 0.0912 | [0.0316, 0.1243] | [0.0074, 0.1750] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s44 | knn_mean_cosine | strict | 0.1016 | 0.1016 | [0.0448, 0.1533] | [0.0183, 0.1850] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitb14_s44 | mahalanobis_l2 | strict | 0.0585 | 0.0585 | [0.0280, 0.0866] | [0.0126, 0.1043] | pass_old_only | pass_both | 30/2 | 2 | 768 | 0.9717 | 0 |
| camelyon_dinov2_vitl14_s42 | knn_mean_cosine | strict | 0.1073 | 0.1073 | [0.0540, 0.1441] | [0.0582, 0.1563] | pass_both | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s42 | mahalanobis_l2 | strict | 0.0741 | 0.0741 | [0.0411, 0.0981] | [0.0415, 0.1066] | pass_both | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s43 | knn_mean_cosine | strict | 0.1258 | 0.1258 | [0.0527, 0.1677] | [0.0170, 0.2347] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s43 | mahalanobis_l2 | strict | 0.0891 | 0.0891 | [0.0365, 0.1207] | [0.0142, 0.1639] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s44 | knn_mean_cosine | strict | 0.0947 | 0.0947 | [0.0434, 0.1432] | [0.0136, 0.1758] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_dinov2_vitl14_s44 | mahalanobis_l2 | strict | 0.0631 | 0.0631 | [0.0292, 0.0967] | [0.0046, 0.1216] | pass_old_only | pass_both | 30/2 | 2 | 1024 | 0.9765 | 0 |
| camelyon_effb3_s42 | knn_mean_cosine | strict | 0.1363 | 0.1363 | [0.0419, 0.1890] | [0.0373, 0.2353] | pass_both | pass_both | 30/2 | 2 | 1536 | 0.9963 | 0 |
| camelyon_effb3_s42 | mahalanobis_l2 | strict | 0.1402 | 0.1402 | [0.0453, 0.1863] | [0.0160, 0.2643] | pass_old_only | pass_both | 30/2 | 2 | 1536 | 0.9963 | 0 |
| camelyon_efficientnet_v2_s_s42 | knn_mean_cosine | strict | 0.0864 | 0.0864 | [0.0406, 0.1197] | [0.0317, 0.1410] | pass_both | pass_both | 30/2 | 2 | 1280 | 0.9963 | 0 |
| camelyon_efficientnet_v2_s_s42 | mahalanobis_l2 | strict | 0.0831 | 0.0831 | [0.0218, 0.1177] | [0.0123, 0.1539] | pass_old_only | pass_both | 30/2 | 2 | 1280 | 0.9963 | 0 |
| camelyon_mobilenet_v3_large_s42 | knn_mean_cosine | strict | 0.0911 | 0.0911 | [0.0512, 0.1268] | [0.0381, 0.1441] | pass_both | pass_both | 30/2 | 2 | 960 | 0.9965 | 0 |
| camelyon_mobilenet_v3_large_s42 | mahalanobis_l2 | strict | 0.0988 | 0.0988 | [0.0385, 0.1428] | [0.0180, 0.1796] | pass_old_only | pass_both | 30/2 | 2 | 960 | 0.9965 | 0 |
| camelyon_regnet_y_3_2gf_s42 | knn_mean_cosine | strict | 0.1223 | 0.1223 | [0.0588, 0.1579] | [0.0342, 0.2103] | pass_both | pass_both | 30/2 | 2 | 1512 | 0.9965 | 0 |
| camelyon_regnet_y_3_2gf_s42 | mahalanobis_l2 | strict | 0.1492 | 0.1492 | [0.0481, 0.2107] | [0.0057, 0.2927] | pass_old_only | pass_both | 30/2 | 2 | 1512 | 0.9965 | 0 |
| camelyon_resnet18_s42 | knn_mean_cosine | strict | 0.1358 | 0.1358 | [0.0437, 0.2071] | [0.0082, 0.2634] | pass_old_only | pass_both | 30/2 | 2 | 512 | 0.9954 | 0 |
| camelyon_resnet18_s42 | mahalanobis_l2 | strict | 0.1683 | 0.1683 | [0.0368, 0.2418] | [-0.0051, 0.3416] | pass_old_only | pass_old_only | 30/2 | 2 | 512 | 0.9954 | 0 |
| camelyon_resnet50_s42 | knn_mean_cosine | strict | 0.2033 | 0.2033 | [0.0711, 0.2746] | [0.0779, 0.3288] | pass_both | pass_both | 30/2 | 2 | 2048 | 0.9954 | 0 |
| camelyon_resnet50_s42 | mahalanobis_l2 | strict | 0.2156 | 0.2156 | [0.0543, 0.2892] | [0.0938, 0.3374] | pass_both | pass_both | 30/2 | 2 | 2048 | 0.9954 | 0 |
| camelyon_resnet50_s43 | knn_mean_cosine | strict | 0.1319 | 0.1319 | [0.0166, 0.2136] | [-0.0515, 0.3152] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9952 | 0 |
| camelyon_resnet50_s43 | mahalanobis_l2 | strict | 0.0776 | 0.0776 | [0.0134, 0.1157] | [-0.0042, 0.1595] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9952 | 0 |
| camelyon_resnet50_s44 | knn_mean_cosine | strict | 0.0902 | 0.0902 | [0.0170, 0.1726] | [-0.0514, 0.2317] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9954 | 0 |
| camelyon_resnet50_s44 | mahalanobis_l2 | strict | 0.0697 | 0.0697 | [0.0082, 0.1398] | [-0.0721, 0.2116] | fail_both | pass_old_only | 30/2 | 2 | 2048 | 0.9954 | 0 |
| camelyon_uni_s42 | knn_mean_cosine | strict | 0.0933 | 0.0933 | [0.0144, 0.1776] | [-0.0207, 0.2073] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s42 | mahalanobis_l2 | strict | 0.0077 | 0.0077 | [0.0021, 0.0132] | [-0.0091, 0.0245] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s43 | knn_mean_cosine | strict | 0.1133 | 0.1133 | [0.0130, 0.1809] | [-0.0410, 0.2676] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s43 | mahalanobis_l2 | strict | 0.0034 | 0.0034 | [0.0007, 0.0062] | [-0.0015, 0.0083] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s44 | knn_mean_cosine | strict | 0.1050 | 0.1050 | [0.0079, 0.2132] | [-0.0490, 0.2591] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_uni_s44 | mahalanobis_l2 | strict | 0.0045 | 0.0045 | [0.0010, 0.0085] | [-0.0023, 0.0113] | fail_both | pass_old_only | 30/2 | 2 | 1024 | 0.9944 | 0 |
| camelyon_virchow2_s42 | knn_mean_cosine | strict | 0.1121 | 0.1121 | [0.0377, 0.1914] | [0.0248, 0.1993] | pass_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s42 | mahalanobis_l2 | strict | 0.0325 | 0.0325 | [0.0141, 0.0524] | [0.0083, 0.0568] | fail_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s43 | knn_mean_cosine | strict | 0.1525 | 0.1525 | [0.0334, 0.2324] | [-0.0195, 0.3245] | pass_old_only | pass_old_only | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s43 | mahalanobis_l2 | strict | 0.0255 | 0.0255 | [0.0086, 0.0375] | [0.0041, 0.0470] | fail_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s44 | knn_mean_cosine | strict | 0.1168 | 0.1168 | [0.0264, 0.2133] | [-0.0194, 0.2530] | pass_old_only | pass_old_only | 30/2 | 2 | 2560 | 0.9948 | 0 |
| camelyon_virchow2_s44 | mahalanobis_l2 | strict | 0.0226 | 0.0226 | [0.0067, 0.0387] | [0.0038, 0.0414] | fail_both | pass_both | 30/2 | 2 | 2560 | 0.9948 | 0 |
| dermamnist_convnext_tiny_s42_std | knn_mean_cosine | strict | 0.0051 | 0.0051 | [0.0037, 0.0066] | [0.0030, 0.0071] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.9235 | 0 |
| dermamnist_convnext_tiny_s42_std | mahalanobis_l2 | strict | 0.0196 | 0.0196 | [0.0160, 0.0232] | [0.0150, 0.0242] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.9235 | 0 |
| dermamnist_convnext_tiny_s43_std | knn_mean_cosine | strict | 0.0057 | 0.0057 | [0.0041, 0.0072] | [0.0035, 0.0078] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.9177 | 0 |
| dermamnist_convnext_tiny_s43_std | mahalanobis_l2 | strict | 0.0258 | 0.0258 | [0.0217, 0.0301] | [0.0215, 0.0302] | pass_both | pass_both | 5678/2 | 2 | 768 | 0.9177 | 0 |
| dermamnist_densenet121_s42_std | knn_mean_cosine | strict | 0.0062 | 0.0062 | [0.0031, 0.0092] | [0.0018, 0.0106] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8745 | 0 |
| dermamnist_densenet121_s42_std | mahalanobis_l2 | strict | 0.0158 | 0.0158 | [0.0118, 0.0200] | [0.0097, 0.0219] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8745 | 0 |
| dermamnist_densenet121_s43_std | knn_mean_cosine | strict | 0.0051 | 0.0051 | [0.0025, 0.0080] | [0.0007, 0.0094] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8716 | 0 |
| dermamnist_densenet121_s43_std | mahalanobis_l2 | strict | 0.0138 | 0.0138 | [0.0098, 0.0185] | [0.0085, 0.0190] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.8716 | 0 |
| dermamnist_effb3_s42_std | knn_mean_cosine | strict | 0.0078 | 0.0078 | [0.0055, 0.0100] | [0.0047, 0.0108] | fail_both | pass_both | 5678/2 | 2 | 1536 | 0.8817 | 0 |
| dermamnist_effb3_s42_std | mahalanobis_l2 | strict | 0.0235 | 0.0235 | [0.0191, 0.0282] | [0.0181, 0.0289] | fail_both | pass_both | 5678/2 | 2 | 1536 | 0.8817 | 0 |
| dermamnist_efficientnet_v2_s_s42_std | knn_mean_cosine | strict | 0.0041 | 0.0041 | [0.0028, 0.0054] | [0.0021, 0.0061] | fail_both | pass_both | 5678/2 | 2 | 1280 | 0.8990 | 0 |
| dermamnist_efficientnet_v2_s_s42_std | mahalanobis_l2 | strict | 0.0141 | 0.0141 | [0.0104, 0.0183] | [0.0092, 0.0190] | fail_both | pass_both | 5678/2 | 2 | 1280 | 0.8990 | 0 |
| dermamnist_fm_conch_v1_5_s42_std | knn_mean_cosine | strict | 0.0031 | 0.0031 | [0.0011, 0.0054] | [-0.0010, 0.0073] | fail_both | pass_old_only | 5678/2 | 2 | 768 | 0.7316 | 0 |
| dermamnist_fm_conch_v1_5_s42_std | mahalanobis_l2 | strict | 0.0079 | 0.0079 | [0.0054, 0.0105] | [0.0041, 0.0117] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.7316 | 0 |
| dermamnist_fm_dinov2_vitb14_s42_std | knn_mean_cosine | strict | 0.0078 | 0.0078 | [0.0059, 0.0098] | [0.0046, 0.0110] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.7172 | 0 |
| dermamnist_fm_dinov2_vitb14_s42_std | mahalanobis_l2 | strict | 0.0126 | 0.0126 | [0.0099, 0.0156] | [0.0094, 0.0158] | fail_both | pass_both | 5678/2 | 2 | 768 | 0.7172 | 0 |
| dermamnist_fm_dinov2_vitl14_s42_std | knn_mean_cosine | strict | 0.0067 | 0.0067 | [0.0048, 0.0087] | [0.0040, 0.0094] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.6782 | 0 |
| dermamnist_fm_dinov2_vitl14_s42_std | mahalanobis_l2 | strict | 0.0142 | 0.0142 | [0.0107, 0.0182] | [0.0103, 0.0181] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.6782 | 0 |
| dermamnist_fm_uni_s42_std | knn_mean_cosine | strict | 0.0028 | 0.0028 | [0.0017, 0.0041] | [0.0010, 0.0046] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.7201 | 0 |
| dermamnist_fm_uni_s42_std | mahalanobis_l2 | strict | 0.0084 | 0.0084 | [0.0054, 0.0115] | [0.0050, 0.0117] | fail_both | pass_both | 5678/2 | 2 | 1024 | 0.7201 | 0 |
| dermamnist_fm_virchow2_s42_std | knn_mean_cosine | strict | 0.0023 | 0.0023 | [0.0009, 0.0037] | [-0.0000, 0.0045] | fail_both | pass_old_only | 5678/2 | 2 | 2560 | 0.7388 | 0 |
| dermamnist_fm_virchow2_s42_std | mahalanobis_l2 | strict | 0.0120 | 0.0120 | [0.0087, 0.0154] | [0.0079, 0.0161] | fail_both | pass_both | 5678/2 | 2 | 2560 | 0.7388 | 0 |
| dermamnist_mobilenet_v3_large_s42_std | knn_mean_cosine | strict | 0.0127 | 0.0127 | [0.0085, 0.0166] | [0.0071, 0.0184] | fail_both | pass_both | 5678/2 | 2 | 960 | 0.8268 | 0 |
| dermamnist_mobilenet_v3_large_s42_std | mahalanobis_l2 | strict | 0.0281 | 0.0281 | [0.0227, 0.0332] | [0.0222, 0.0340] | pass_both | pass_both | 5678/2 | 2 | 960 | 0.8268 | 0 |
| dermamnist_regnet_y_3_2gf_s42_std | knn_mean_cosine | strict | 0.0114 | 0.0114 | [0.0080, 0.0150] | [0.0063, 0.0166] | fail_both | pass_both | 5678/2 | 2 | 1512 | 0.8672 | 0 |
| dermamnist_regnet_y_3_2gf_s42_std | mahalanobis_l2 | strict | 0.0292 | 0.0292 | [0.0237, 0.0354] | [0.0213, 0.0372] | pass_both | pass_both | 5678/2 | 2 | 1512 | 0.8672 | 0 |
| dermamnist_resnet18_s42_std | knn_mean_cosine | strict | 0.0065 | 0.0065 | [0.0026, 0.0098] | [-0.0001, 0.0130] | fail_both | pass_old_only | 5678/2 | 2 | 512 | 0.8384 | 0 |
| dermamnist_resnet18_s42_std | mahalanobis_l2 | strict | 0.0214 | 0.0214 | [0.0165, 0.0265] | [0.0150, 0.0279] | fail_both | pass_both | 5678/2 | 2 | 512 | 0.8384 | 0 |
| dermamnist_resnet18_s43_std | knn_mean_cosine | strict | 0.0100 | 0.0100 | [0.0062, 0.0141] | [0.0044, 0.0156] | fail_both | pass_both | 5678/2 | 2 | 512 | 0.8398 | 0 |
| dermamnist_resnet18_s43_std | mahalanobis_l2 | strict | 0.0285 | 0.0285 | [0.0230, 0.0343] | [0.0206, 0.0364] | pass_both | pass_both | 5678/2 | 2 | 512 | 0.8398 | 0 |
| dermamnist_resnet50_s42_std | knn_mean_cosine | strict | 0.0047 | 0.0047 | [0.0016, 0.0079] | [-0.0009, 0.0103] | fail_both | pass_old_only | 5678/2 | 2 | 2048 | 0.8470 | 0 |
| dermamnist_resnet50_s42_std | mahalanobis_l2 | strict | 0.0206 | 0.0206 | [0.0168, 0.0244] | [0.0152, 0.0259] | fail_both | pass_both | 5678/2 | 2 | 2048 | 0.8470 | 0 |
| dermamnist_resnet50_s43_std | knn_mean_cosine | strict | 0.0063 | 0.0063 | [0.0042, 0.0084] | [0.0028, 0.0098] | fail_both | pass_both | 5678/2 | 2 | 2048 | 0.8716 | 0 |
| dermamnist_resnet50_s43_std | mahalanobis_l2 | strict | 0.0181 | 0.0181 | [0.0142, 0.0224] | [0.0127, 0.0235] | fail_both | pass_both | 5678/2 | 2 | 2048 | 0.8716 | 0 |
| isic2019_convnext_tiny_s42_std | knn_mean_cosine | strict | 0.0204 | 0.0194 | [0.0183, 0.0224] | [0.0179, 0.0228] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 0 |
| isic2019_convnext_tiny_s42_std | knn_mean_cosine | tracka | 0.0194 | 0.0194 | [0.0174, 0.0214] | [0.0169, 0.0219] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 117 |
| isic2019_convnext_tiny_s42_std | mahalanobis_l2 | strict | 0.0443 | 0.0426 | [0.0406, 0.0480] | [0.0404, 0.0482] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 0 |
| isic2019_convnext_tiny_s42_std | mahalanobis_l2 | tracka | 0.0426 | 0.0426 | [0.0391, 0.0462] | [0.0386, 0.0465] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8636 | 117 |
| isic2019_convnext_tiny_s43_std | knn_mean_cosine | strict | 0.0196 | 0.0188 | [0.0176, 0.0215] | [0.0171, 0.0221] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 0 |
| isic2019_convnext_tiny_s43_std | knn_mean_cosine | tracka | 0.0188 | 0.0188 | [0.0167, 0.0207] | [0.0161, 0.0214] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 117 |
| isic2019_convnext_tiny_s43_std | mahalanobis_l2 | strict | 0.0458 | 0.0441 | [0.0421, 0.0495] | [0.0418, 0.0497] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 0 |
| isic2019_convnext_tiny_s43_std | mahalanobis_l2 | tracka | 0.0441 | 0.0441 | [0.0405, 0.0478] | [0.0398, 0.0483] | pass_both | pass_both | 11041/2 | 2 | 768 | 0.8722 | 117 |
| isic2019_densenet121_s42_std | knn_mean_cosine | strict | 0.0245 | 0.0234 | [0.0214, 0.0273] | [0.0211, 0.0279] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 0 |
| isic2019_densenet121_s42_std | knn_mean_cosine | tracka | 0.0234 | 0.0234 | [0.0207, 0.0264] | [0.0198, 0.0270] | pass_old_only | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 117 |
| isic2019_densenet121_s42_std | mahalanobis_l2 | strict | 0.0358 | 0.0345 | [0.0324, 0.0394] | [0.0318, 0.0398] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 0 |
| isic2019_densenet121_s42_std | mahalanobis_l2 | tracka | 0.0345 | 0.0345 | [0.0315, 0.0379] | [0.0305, 0.0386] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8150 | 117 |
| isic2019_densenet121_s43_std | knn_mean_cosine | strict | 0.0259 | 0.0247 | [0.0227, 0.0290] | [0.0215, 0.0303] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 0 |
| isic2019_densenet121_s43_std | knn_mean_cosine | tracka | 0.0247 | 0.0247 | [0.0218, 0.0276] | [0.0209, 0.0284] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 117 |
| isic2019_densenet121_s43_std | mahalanobis_l2 | strict | 0.0370 | 0.0356 | [0.0335, 0.0403] | [0.0333, 0.0407] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 0 |
| isic2019_densenet121_s43_std | mahalanobis_l2 | tracka | 0.0356 | 0.0356 | [0.0324, 0.0389] | [0.0318, 0.0394] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.8037 | 117 |
| isic2019_effb3_s42_std | knn_mean_cosine | strict | 0.0179 | 0.0172 | [0.0159, 0.0199] | [0.0151, 0.0207] | fail_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 0 |
| isic2019_effb3_s42_std | knn_mean_cosine | tracka | 0.0172 | 0.0172 | [0.0153, 0.0192] | [0.0145, 0.0199] | fail_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 117 |
| isic2019_effb3_s42_std | mahalanobis_l2 | strict | 0.0475 | 0.0458 | [0.0438, 0.0516] | [0.0434, 0.0516] | pass_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 0 |
| isic2019_effb3_s42_std | mahalanobis_l2 | tracka | 0.0458 | 0.0458 | [0.0423, 0.0494] | [0.0419, 0.0498] | pass_both | pass_both | 11041/2 | 2 | 1536 | 0.8325 | 117 |
| isic2019_efficientnet_v2_s_s42_std | knn_mean_cosine | strict | 0.0092 | 0.0089 | [0.0079, 0.0107] | [0.0075, 0.0109] | fail_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 0 |
| isic2019_efficientnet_v2_s_s42_std | knn_mean_cosine | tracka | 0.0089 | 0.0089 | [0.0076, 0.0102] | [0.0073, 0.0105] | fail_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 117 |
| isic2019_efficientnet_v2_s_s42_std | mahalanobis_l2 | strict | 0.0344 | 0.0333 | [0.0316, 0.0374] | [0.0314, 0.0375] | pass_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 0 |
| isic2019_efficientnet_v2_s_s42_std | mahalanobis_l2 | tracka | 0.0333 | 0.0333 | [0.0307, 0.0361] | [0.0302, 0.0364] | pass_both | pass_both | 11041/2 | 2 | 1280 | 0.8769 | 117 |
| isic2019_fm_conch_v1_5_s42_std | knn_mean_cosine | strict | 0.0170 | 0.0162 | [0.0149, 0.0191] | [0.0141, 0.0198] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 0 |
| isic2019_fm_conch_v1_5_s42_std | knn_mean_cosine | tracka | 0.0162 | 0.0162 | [0.0141, 0.0184] | [0.0131, 0.0192] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 117 |
| isic2019_fm_conch_v1_5_s42_std | mahalanobis_l2 | strict | 0.0176 | 0.0172 | [0.0160, 0.0193] | [0.0159, 0.0194] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 0 |
| isic2019_fm_conch_v1_5_s42_std | mahalanobis_l2 | tracka | 0.0172 | 0.0172 | [0.0156, 0.0188] | [0.0153, 0.0191] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6829 | 117 |
| isic2019_fm_dinov2_vitb14_s42_std | knn_mean_cosine | strict | 0.0131 | 0.0126 | [0.0116, 0.0147] | [0.0110, 0.0152] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 0 |
| isic2019_fm_dinov2_vitb14_s42_std | knn_mean_cosine | tracka | 0.0126 | 0.0126 | [0.0110, 0.0143] | [0.0105, 0.0147] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 117 |
| isic2019_fm_dinov2_vitb14_s42_std | mahalanobis_l2 | strict | 0.0209 | 0.0202 | [0.0193, 0.0226] | [0.0190, 0.0228] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 0 |
| isic2019_fm_dinov2_vitb14_s42_std | mahalanobis_l2 | tracka | 0.0202 | 0.0202 | [0.0186, 0.0219] | [0.0182, 0.0223] | fail_both | pass_both | 11041/2 | 2 | 768 | 0.6888 | 117 |
| isic2019_fm_dinov2_vitl14_s42_std | knn_mean_cosine | strict | 0.0134 | 0.0129 | [0.0117, 0.0149] | [0.0115, 0.0153] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 0 |
| isic2019_fm_dinov2_vitl14_s42_std | knn_mean_cosine | tracka | 0.0129 | 0.0129 | [0.0113, 0.0146] | [0.0109, 0.0149] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 117 |
| isic2019_fm_dinov2_vitl14_s42_std | mahalanobis_l2 | strict | 0.0275 | 0.0265 | [0.0255, 0.0294] | [0.0249, 0.0301] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 0 |
| isic2019_fm_dinov2_vitl14_s42_std | mahalanobis_l2 | tracka | 0.0265 | 0.0265 | [0.0245, 0.0284] | [0.0241, 0.0288] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6975 | 117 |
| isic2019_fm_uni_s42_std | knn_mean_cosine | strict | 0.0182 | 0.0175 | [0.0161, 0.0201] | [0.0158, 0.0205] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 0 |
| isic2019_fm_uni_s42_std | knn_mean_cosine | tracka | 0.0175 | 0.0175 | [0.0156, 0.0193] | [0.0146, 0.0204] | fail_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 117 |
| isic2019_fm_uni_s42_std | mahalanobis_l2 | strict | 0.0369 | 0.0355 | [0.0342, 0.0397] | [0.0341, 0.0398] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 0 |
| isic2019_fm_uni_s42_std | mahalanobis_l2 | tracka | 0.0355 | 0.0355 | [0.0330, 0.0380] | [0.0323, 0.0387] | pass_both | pass_both | 11041/2 | 2 | 1024 | 0.6845 | 117 |
| isic2019_fm_virchow2_s42_std | knn_mean_cosine | strict | 0.0128 | 0.0126 | [0.0110, 0.0146] | [0.0105, 0.0151] | fail_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 0 |
| isic2019_fm_virchow2_s42_std | knn_mean_cosine | tracka | 0.0126 | 0.0126 | [0.0107, 0.0142] | [0.0095, 0.0157] | fail_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 117 |
| isic2019_fm_virchow2_s42_std | mahalanobis_l2 | strict | 0.0527 | 0.0508 | [0.0490, 0.0563] | [0.0489, 0.0566] | pass_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 0 |
| isic2019_fm_virchow2_s42_std | mahalanobis_l2 | tracka | 0.0508 | 0.0508 | [0.0473, 0.0543] | [0.0466, 0.0549] | pass_both | pass_both | 11041/2 | 2 | 2560 | 0.7355 | 117 |
| isic2019_mobilenet_v3_large_s42_std | knn_mean_cosine | strict | 0.0212 | 0.0203 | [0.0186, 0.0237] | [0.0179, 0.0245] | fail_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 0 |
| isic2019_mobilenet_v3_large_s42_std | knn_mean_cosine | tracka | 0.0203 | 0.0203 | [0.0181, 0.0228] | [0.0168, 0.0238] | fail_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 117 |
| isic2019_mobilenet_v3_large_s42_std | mahalanobis_l2 | strict | 0.0373 | 0.0358 | [0.0343, 0.0403] | [0.0339, 0.0407] | pass_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 0 |
| isic2019_mobilenet_v3_large_s42_std | mahalanobis_l2 | tracka | 0.0358 | 0.0358 | [0.0330, 0.0386] | [0.0322, 0.0394] | pass_both | pass_both | 11041/2 | 2 | 960 | 0.7514 | 117 |
| isic2019_regnet_y_3_2gf_s42_std | knn_mean_cosine | strict | 0.0312 | 0.0299 | [0.0278, 0.0344] | [0.0270, 0.0354] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 0 |
| isic2019_regnet_y_3_2gf_s42_std | knn_mean_cosine | tracka | 0.0299 | 0.0299 | [0.0268, 0.0329] | [0.0262, 0.0336] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 117 |
| isic2019_regnet_y_3_2gf_s42_std | mahalanobis_l2 | strict | 0.0694 | 0.0667 | [0.0640, 0.0750] | [0.0635, 0.0752] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 0 |
| isic2019_regnet_y_3_2gf_s42_std | mahalanobis_l2 | tracka | 0.0667 | 0.0667 | [0.0616, 0.0719] | [0.0608, 0.0725] | pass_both | pass_both | 11041/2 | 2 | 1512 | 0.8011 | 117 |
| isic2019_resnet18_s42_std | knn_mean_cosine | strict | 0.0207 | 0.0196 | [0.0182, 0.0234] | [0.0170, 0.0245] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 0 |
| isic2019_resnet18_s42_std | knn_mean_cosine | tracka | 0.0196 | 0.0196 | [0.0172, 0.0221] | [0.0161, 0.0232] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 117 |
| isic2019_resnet18_s42_std | mahalanobis_l2 | strict | 0.0284 | 0.0270 | [0.0258, 0.0311] | [0.0247, 0.0322] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 0 |
| isic2019_resnet18_s42_std | mahalanobis_l2 | tracka | 0.0270 | 0.0270 | [0.0245, 0.0297] | [0.0236, 0.0305] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7795 | 117 |
| isic2019_resnet18_s43_std | knn_mean_cosine | strict | 0.0219 | 0.0207 | [0.0195, 0.0245] | [0.0185, 0.0254] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 0 |
| isic2019_resnet18_s43_std | knn_mean_cosine | tracka | 0.0207 | 0.0207 | [0.0183, 0.0232] | [0.0175, 0.0239] | fail_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 117 |
| isic2019_resnet18_s43_std | mahalanobis_l2 | strict | 0.0312 | 0.0300 | [0.0285, 0.0343] | [0.0270, 0.0354] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 0 |
| isic2019_resnet18_s43_std | mahalanobis_l2 | tracka | 0.0300 | 0.0300 | [0.0270, 0.0327] | [0.0265, 0.0335] | pass_both | pass_both | 11041/2 | 2 | 512 | 0.7759 | 117 |
| isic2019_resnet50_s42_std | knn_mean_cosine | strict | 0.0154 | 0.0146 | [0.0129, 0.0179] | [0.0123, 0.0185] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 0 |
| isic2019_resnet50_s42_std | knn_mean_cosine | tracka | 0.0146 | 0.0146 | [0.0123, 0.0169] | [0.0111, 0.0180] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 117 |
| isic2019_resnet50_s42_std | mahalanobis_l2 | strict | 0.0600 | 0.0575 | [0.0546, 0.0658] | [0.0540, 0.0659] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 0 |
| isic2019_resnet50_s42_std | mahalanobis_l2 | tracka | 0.0575 | 0.0575 | [0.0529, 0.0628] | [0.0523, 0.0627] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8054 | 117 |
| isic2019_resnet50_s43_std | knn_mean_cosine | strict | 0.0177 | 0.0167 | [0.0153, 0.0201] | [0.0140, 0.0213] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 0 |
| isic2019_resnet50_s43_std | knn_mean_cosine | tracka | 0.0167 | 0.0167 | [0.0145, 0.0192] | [0.0140, 0.0195] | fail_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 117 |
| isic2019_resnet50_s43_std | mahalanobis_l2 | strict | 0.0578 | 0.0556 | [0.0527, 0.0631] | [0.0518, 0.0638] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 0 |
| isic2019_resnet50_s43_std | mahalanobis_l2 | tracka | 0.0556 | 0.0556 | [0.0509, 0.0604] | [0.0502, 0.0610] | pass_both | pass_both | 11041/2 | 2 | 2048 | 0.8034 | 117 |
| kermany_convnext_tiny_s42_std | knn_mean_cosine | strict | 0.0359 | 0.0358 | [0.0276, 0.0443] | [0.0275, 0.0442] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s42_std | knn_mean_cosine | tracka | 0.0358 | 0.0358 | [0.0280, 0.0442] | [0.0270, 0.0446] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_convnext_tiny_s42_std | mahalanobis_l2 | strict | 0.0278 | 0.0277 | [0.0217, 0.0344] | [0.0214, 0.0342] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s42_std | mahalanobis_l2 | tracka | 0.0277 | 0.0277 | [0.0218, 0.0341] | [0.0219, 0.0336] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_convnext_tiny_s43_std | knn_mean_cosine | strict | 0.0434 | 0.0433 | [0.0355, 0.0511] | [0.0356, 0.0511] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s43_std | knn_mean_cosine | tracka | 0.0433 | 0.0433 | [0.0353, 0.0504] | [0.0356, 0.0509] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_convnext_tiny_s43_std | mahalanobis_l2 | strict | 0.0291 | 0.0291 | [0.0232, 0.0353] | [0.0232, 0.0350] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 0 |
| kermany_convnext_tiny_s43_std | mahalanobis_l2 | tracka | 0.0291 | 0.0291 | [0.0230, 0.0354] | [0.0232, 0.0349] | pass_both | pass_both | 4240/2 | 2 | 768 | 0.9985 | 2 |
| kermany_densenet121_s42_std | knn_mean_cosine | strict | 0.0317 | 0.0316 | [0.0247, 0.0385] | [0.0242, 0.0391] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 0 |
| kermany_densenet121_s42_std | knn_mean_cosine | tracka | 0.0316 | 0.0316 | [0.0245, 0.0383] | [0.0237, 0.0395] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 2 |
| kermany_densenet121_s42_std | mahalanobis_l2 | strict | 0.0287 | 0.0286 | [0.0231, 0.0344] | [0.0225, 0.0349] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 0 |
| kermany_densenet121_s42_std | mahalanobis_l2 | tracka | 0.0286 | 0.0286 | [0.0230, 0.0340] | [0.0218, 0.0355] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9940 | 2 |
| kermany_densenet121_s43_std | knn_mean_cosine | strict | 0.0297 | 0.0296 | [0.0242, 0.0350] | [0.0228, 0.0366] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 0 |
| kermany_densenet121_s43_std | knn_mean_cosine | tracka | 0.0296 | 0.0296 | [0.0243, 0.0351] | [0.0221, 0.0372] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 2 |
| kermany_densenet121_s43_std | mahalanobis_l2 | strict | 0.0246 | 0.0246 | [0.0205, 0.0290] | [0.0201, 0.0291] | pass_both | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 0 |
| kermany_densenet121_s43_std | mahalanobis_l2 | tracka | 0.0246 | 0.0246 | [0.0204, 0.0290] | [0.0191, 0.0301] | pass_old_only | pass_both | 4240/2 | 2 | 1024 | 0.9955 | 2 |
| kermany_effb3_s42_std | knn_mean_cosine | strict | 0.0211 | 0.0210 | [0.0168, 0.0256] | [0.0165, 0.0256] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 0 |
| kermany_effb3_s42_std | knn_mean_cosine | tracka | 0.0210 | 0.0210 | [0.0168, 0.0256] | [0.0164, 0.0257] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 2 |
| kermany_effb3_s42_std | mahalanobis_l2 | strict | 0.0205 | 0.0204 | [0.0172, 0.0235] | [0.0171, 0.0238] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 0 |
| kermany_effb3_s42_std | mahalanobis_l2 | tracka | 0.0204 | 0.0204 | [0.0172, 0.0237] | [0.0172, 0.0236] | fail_both | pass_both | 4240/2 | 2 | 1536 | 0.9940 | 2 |
| kermany_efficientnet_v2_s_s42_std | knn_mean_cosine | strict | 0.0187 | 0.0186 | [0.0149, 0.0229] | [0.0145, 0.0229] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 0 |
| kermany_efficientnet_v2_s_s42_std | knn_mean_cosine | tracka | 0.0186 | 0.0186 | [0.0148, 0.0228] | [0.0137, 0.0236] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 2 |
| kermany_efficientnet_v2_s_s42_std | mahalanobis_l2 | strict | 0.0232 | 0.0232 | [0.0189, 0.0275] | [0.0191, 0.0273] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 0 |
| kermany_efficientnet_v2_s_s42_std | mahalanobis_l2 | tracka | 0.0232 | 0.0232 | [0.0189, 0.0277] | [0.0183, 0.0280] | fail_both | pass_both | 4240/2 | 2 | 1280 | 0.9970 | 2 |
| kermany_mobilenet_v3_large_s42_std | knn_mean_cosine | strict | 0.0293 | 0.0293 | [0.0234, 0.0355] | [0.0234, 0.0351] | pass_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 0 |
| kermany_mobilenet_v3_large_s42_std | knn_mean_cosine | tracka | 0.0293 | 0.0293 | [0.0234, 0.0353] | [0.0226, 0.0359] | pass_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 2 |
| kermany_mobilenet_v3_large_s42_std | mahalanobis_l2 | strict | 0.0208 | 0.0208 | [0.0171, 0.0245] | [0.0175, 0.0241] | fail_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 0 |
| kermany_mobilenet_v3_large_s42_std | mahalanobis_l2 | tracka | 0.0208 | 0.0208 | [0.0172, 0.0247] | [0.0167, 0.0248] | fail_both | pass_both | 4240/2 | 2 | 960 | 0.9940 | 2 |
| kermany_regnet_y_3_2gf_s42_std | knn_mean_cosine | strict | 0.0628 | 0.0626 | [0.0501, 0.0759] | [0.0488, 0.0769] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 0 |
| kermany_regnet_y_3_2gf_s42_std | knn_mean_cosine | tracka | 0.0626 | 0.0626 | [0.0500, 0.0746] | [0.0469, 0.0784] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 2 |
| kermany_regnet_y_3_2gf_s42_std | mahalanobis_l2 | strict | 0.0532 | 0.0530 | [0.0452, 0.0613] | [0.0435, 0.0628] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 0 |
| kermany_regnet_y_3_2gf_s42_std | mahalanobis_l2 | tracka | 0.0530 | 0.0530 | [0.0447, 0.0612] | [0.0440, 0.0619] | pass_both | pass_both | 4240/2 | 2 | 1512 | 0.9955 | 2 |
| kermany_resnet18_s42_std | knn_mean_cosine | strict | 0.0195 | 0.0195 | [0.0151, 0.0238] | [0.0139, 0.0252] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 0 |
| kermany_resnet18_s42_std | knn_mean_cosine | tracka | 0.0195 | 0.0195 | [0.0150, 0.0235] | [0.0138, 0.0251] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 2 |
| kermany_resnet18_s42_std | mahalanobis_l2 | strict | 0.0166 | 0.0165 | [0.0131, 0.0199] | [0.0124, 0.0208] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 0 |
| kermany_resnet18_s42_std | mahalanobis_l2 | tracka | 0.0165 | 0.0165 | [0.0130, 0.0197] | [0.0120, 0.0211] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9970 | 2 |
| kermany_resnet18_s43_std | knn_mean_cosine | strict | 0.0263 | 0.0262 | [0.0212, 0.0315] | [0.0189, 0.0336] | pass_old_only | pass_both | 4240/2 | 2 | 512 | 0.9985 | 0 |
| kermany_resnet18_s43_std | knn_mean_cosine | tracka | 0.0262 | 0.0262 | [0.0208, 0.0319] | [0.0200, 0.0324] | pass_both | pass_both | 4240/2 | 2 | 512 | 0.9985 | 2 |
| kermany_resnet18_s43_std | mahalanobis_l2 | strict | 0.0243 | 0.0242 | [0.0196, 0.0288] | [0.0193, 0.0293] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9985 | 0 |
| kermany_resnet18_s43_std | mahalanobis_l2 | tracka | 0.0242 | 0.0242 | [0.0198, 0.0287] | [0.0189, 0.0296] | fail_both | pass_both | 4240/2 | 2 | 512 | 0.9985 | 2 |
| kermany_resnet50_s42_std | knn_mean_cosine | strict | 0.0205 | 0.0205 | [0.0154, 0.0253] | [0.0138, 0.0271] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 0 |
| kermany_resnet50_s42_std | knn_mean_cosine | tracka | 0.0205 | 0.0205 | [0.0156, 0.0248] | [0.0139, 0.0271] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 2 |
| kermany_resnet50_s42_std | mahalanobis_l2 | strict | 0.0446 | 0.0445 | [0.0385, 0.0510] | [0.0377, 0.0516] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 0 |
| kermany_resnet50_s42_std | mahalanobis_l2 | tracka | 0.0445 | 0.0445 | [0.0376, 0.0504] | [0.0374, 0.0516] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9940 | 2 |
| kermany_resnet50_s43_std | knn_mean_cosine | strict | 0.0182 | 0.0182 | [0.0146, 0.0221] | [0.0138, 0.0227] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 0 |
| kermany_resnet50_s43_std | knn_mean_cosine | tracka | 0.0182 | 0.0182 | [0.0147, 0.0221] | [0.0129, 0.0235] | fail_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 2 |
| kermany_resnet50_s43_std | mahalanobis_l2 | strict | 0.0416 | 0.0415 | [0.0360, 0.0473] | [0.0367, 0.0465] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 0 |
| kermany_resnet50_s43_std | mahalanobis_l2 | tracka | 0.0415 | 0.0415 | [0.0358, 0.0468] | [0.0359, 0.0471] | pass_both | pass_both | 4240/2 | 2 | 2048 | 0.9955 | 2 |

## Caveats

1. Scorers are the Track A definitions (mahalanobis_l2 = Ledoit-Wolf float64 Mahalanobis; knn_mean_cosine = mean cosine distance to 50 neighbours). Package-kNN numbers are not used. ViM is not reported: Track A ViM is not reproduced within 0.001 (match table).
2. seen = tracka rows (ISIC 2019, Kermany) count orphan seen groups (no training image after the val carve-out) as jackknife units, as Track A does; coverage (item 2) is verified under the strict definition only.
3. Camelyon knn_mean_cosine ran on GPU after passing the strict equality check against sklearn (CPU); all other rows CPU.
4. Technical rerun: the first submission of the medical cells failed at start (a GPU script path leaked into the CPU jobs' environment); rerun unchanged on CPU (jobs 64813 / 64814).
5. n_groups_fit is per fold: the scorer is fit on one of K halves of the n_groups_train training groups.
