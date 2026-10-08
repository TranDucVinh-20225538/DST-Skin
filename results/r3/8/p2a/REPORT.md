# R3 item 8 / P2-a (CXR) — report

Commit: `362a78a`

**Verdict (preregistered gate):** mahalanobis_l2: INCONCLUSIVE (2 gate-passing backbone(s) < 3); knn_mean_cosine: INCONCLUSIVE (2 gate-passing backbone(s) < 3).
**Verdict (post hoc, AUROC-only competence gate):** mahalanobis_l2: leakage not shown (0/4 gate-passing backbones with Delta_fit > 0.02; not robust: 0 with jackknife CI > 0); knn_mean_cosine: leakage not shown (1/4 gate-passing backbones with Delta_fit > 0.02; not robust: 1 with jackknife CI > 0).

## Phase 0 (leak audit)

- M_std = `wang2017` (precommitted rule). G_leak: leak rate 0.820 >= 0.05 -> continue to training.
- Audit sanity: 112,120 images, 30,805 patients, patient-ID parse rate 1.000, official train_val / test patient overlap 0.
- **Finding: the ChestMNIST split is patient-disjoint** (train->test leak 0.0000, train->val 0.0003; mapping accept rate 1.000, label agreement 0.9983). Its test set overlaps the official test list for only 23.2% of images, so it is a different patient-level split, not the official one.
- Wang et al. 2017 image-level 70/10/20 split (seed 0, stratified on No Finding): train->test leak 0.820.

## Sets

| set | images | patients |
|---|---:|---:|
| fit | 70,666 | 22,625 |
| ckpt | 7,818 | 2,514 |
| seen | 16,497 | 6,990 |
| unseen | 4,045 | 3,814 |
| test_ckpt | 1,882 | 809 |
| wang_val | 11,212 | 7,232 |

ckpt = all Wang-train images of 10% of the Wang-train patients (checkpoint selection only); test_ckpt = Wang-test images of those patients (neither seen nor unseen, excluded). Leak rate on the evaluated test images: 0.803. A-fit folds (K = 2): {'1': 11412, '0': 11213} patients. G_floor: unseen 4,045 images / 3,814 patients (>= 200 / 20: pass).

## Competence gate

| run | ID metric | ID_seen | ID_unseen | loss epoch 1 -> final | loss halved | preregistered gate | post hoc (AUROC only) |
|---|---|---:|---:|---|---|---|---|
| resnet50_s42 | CNN macro AUROC | 0.8251 | 0.8128 | 0.1730 -> 0.1477 | no | FAIL | pass |
| resnet50_s43 | CNN macro AUROC | 0.8258 | 0.8156 | 0.1729 -> 0.1477 | no | FAIL | pass |
| convnext_tiny_s42 | CNN macro AUROC | 0.8427 | 0.8358 | 0.1690 -> 0.1382 | no | FAIL | pass |
| convnext_tiny_s43 | CNN macro AUROC | 0.8428 | 0.8359 | 0.1689 -> 0.1376 | no | FAIL | pass |
| dinov2_vitb14 | probe macro AUROC | 0.7662 | 0.7412 | n/a (frozen FM) | n/a | pass | pass |
| rad_dino | probe macro AUROC | 0.8325 | 0.8068 | n/a (frozen FM) | n/a | pass | pass |

Backbone passes if all its runs pass. Preregistered: resnet50 fail, convnext_tiny fail, dinov2_vitb14 pass, rad_dino pass. Post hoc: resnet50 pass, convnext_tiny pass, dinov2_vitb14 pass, rad_dino pass.

## Delta_fit and A-gap — shenzhen (primary OOD (verdict))

| run | scorer | d | K | n_groups_fit (fold 0 / 1) | ID_seen / ID_unseen | leaky | F2 | **Delta_fit** | jackknife 95% CI | bootstrap 95% CI (ref) | truth | A_gap | A_gap_cm | flags |
|---|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| resnet50_s42 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.3592 | 0.3453 | **+0.0139** | [+0.0123, +0.0154] | [+0.0125, +0.0152] | 0.5861 | -0.2269 | -0.1223 | below_chance |
| resnet50_s42 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.3897 | 0.3718 | **+0.0179** | [+0.0162, +0.0196] | [+0.0164, +0.0195] | 0.6050 | -0.2153 | -0.1189 | below_chance |
| resnet50_s43 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.3763 | 0.3604 | **+0.0159** | [+0.0143, +0.0174] | [+0.0144, +0.0172] | 0.6047 | -0.2284 | -0.1220 | below_chance |
| resnet50_s43 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.4189 | 0.3995 | **+0.0194** | [+0.0177, +0.0210] | [+0.0179, +0.0209] | 0.6345 | -0.2156 | -0.1190 | below_chance |
| convnext_tiny_s42 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.4782 | 0.4485 | **+0.0297** | [+0.0276, +0.0317] | [+0.0278, +0.0315] | 0.6433 | -0.1651 | -0.1250 | below_chance |
| convnext_tiny_s42 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.5000 | 0.4685 | **+0.0315** | [+0.0289, +0.0342] | [+0.0288, +0.0345] | 0.6373 | -0.1373 | -0.1115 | - |
| convnext_tiny_s43 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.4283 | 0.4057 | **+0.0227** | [+0.0210, +0.0243] | [+0.0211, +0.0241] | 0.6087 | -0.1803 | -0.1298 | below_chance |
| convnext_tiny_s43 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.4906 | 0.4646 | **+0.0260** | [+0.0240, +0.0280] | [+0.0239, +0.0283] | 0.6508 | -0.1603 | -0.1233 | below_chance |
| dinov2_vitb14 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.5164 | 0.5009 | **+0.0155** | [+0.0140, +0.0169] | [+0.0147, +0.0163] | 0.6875 | -0.1711 | -0.0586 | - |
| dinov2_vitb14 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.4103 | 0.3948 | **+0.0154** | [+0.0135, +0.0174] | [+0.0147, +0.0162] | 0.6190 | -0.2087 | -0.0708 | below_chance |
| rad_dino | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9820 | 0.9607 | **+0.0214** | [+0.0179, +0.0249] | [+0.0177, +0.0252] | 0.9830 | -0.0009 | +0.0074 | ceiling |
| rad_dino | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9414 | 0.8521 | **+0.0893** | [+0.0814, +0.0972] | [+0.0810, +0.0975] | 0.9435 | -0.0022 | +0.0222 | - |

## Delta_fit and A-gap — kermany_ped (secondary OOD)

| run | scorer | d | K | n_groups_fit (fold 0 / 1) | ID_seen / ID_unseen | leaky | F2 | **Delta_fit** | jackknife 95% CI | bootstrap 95% CI (ref) | truth | A_gap | A_gap_cm | flags |
|---|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| resnet50_s42 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.8324 | 0.8186 | **+0.0138** | [+0.0123, +0.0153] | [+0.0127, +0.0150] | 0.9213 | -0.0890 | -0.0492 | - |
| resnet50_s42 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.8541 | 0.8383 | **+0.0158** | [+0.0143, +0.0174] | [+0.0146, +0.0171] | 0.9269 | -0.0728 | -0.0421 | - |
| resnet50_s43 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.8206 | 0.8078 | **+0.0128** | [+0.0111, +0.0145] | [+0.0117, +0.0140] | 0.9137 | -0.0931 | -0.0531 | - |
| resnet50_s43 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.8319 | 0.8192 | **+0.0127** | [+0.0111, +0.0142] | [+0.0115, +0.0138] | 0.9146 | -0.0827 | -0.0483 | - |
| convnext_tiny_s42 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.8961 | 0.8903 | **+0.0058** | [+0.0048, +0.0067] | [+0.0050, +0.0066] | 0.9367 | -0.0406 | -0.0396 | - |
| convnext_tiny_s42 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.8850 | 0.8713 | **+0.0137** | [+0.0123, +0.0151] | [+0.0123, +0.0152] | 0.9243 | -0.0393 | -0.0405 | - |
| convnext_tiny_s43 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.8363 | 0.8278 | **+0.0085** | [+0.0071, +0.0098] | [+0.0075, +0.0095] | 0.9094 | -0.0731 | -0.0617 | - |
| convnext_tiny_s43 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.8609 | 0.8455 | **+0.0154** | [+0.0136, +0.0172] | [+0.0136, +0.0174] | 0.9210 | -0.0601 | -0.0553 | - |
| dinov2_vitb14 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.9378 | 0.9328 | **+0.0051** | [+0.0045, +0.0056] | [+0.0046, +0.0055] | 0.9697 | -0.0319 | -0.0029 | - |
| dinov2_vitb14 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.9002 | 0.8907 | **+0.0095** | [+0.0087, +0.0103] | [+0.0088, +0.0103] | 0.9615 | -0.0613 | -0.0099 | - |
| rad_dino | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9916 | 0.9853 | **+0.0063** | [+0.0044, +0.0083] | [+0.0050, +0.0078] | 0.9909 | +0.0008 | +0.0038 | ceiling |
| rad_dino | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9261 | 0.8363 | **+0.0897** | [+0.0643, +0.1152] | [+0.0832, +0.0960] | 0.9314 | -0.0053 | +0.0288 | - |

## Delta_fit and A-gap — shenzhen_tb (descriptive, Shenzhen TB subset)

| run | scorer | d | K | n_groups_fit (fold 0 / 1) | ID_seen / ID_unseen | leaky | F2 | **Delta_fit** | jackknife 95% CI | bootstrap 95% CI (ref) | truth | A_gap | A_gap_cm | flags |
|---|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| resnet50_s42 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.5088 | 0.4895 | **+0.0193** | [+0.0170, +0.0215] | [+0.0174, +0.0211] | 0.7285 | -0.2198 | -0.1182 | - |
| resnet50_s42 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.5391 | 0.5174 | **+0.0217** | [+0.0191, +0.0242] | [+0.0195, +0.0237] | 0.7354 | -0.1963 | -0.1094 | - |
| resnet50_s43 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.5114 | 0.4920 | **+0.0194** | [+0.0172, +0.0217] | [+0.0176, +0.0213] | 0.7233 | -0.2119 | -0.1141 | - |
| resnet50_s43 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.5471 | 0.5271 | **+0.0199** | [+0.0176, +0.0223] | [+0.0179, +0.0218] | 0.7394 | -0.1924 | -0.1071 | - |
| convnext_tiny_s42 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.5976 | 0.5683 | **+0.0293** | [+0.0267, +0.0319] | [+0.0269, +0.0316] | 0.7475 | -0.1499 | -0.1211 | - |
| convnext_tiny_s42 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.6475 | 0.6171 | **+0.0304** | [+0.0271, +0.0337] | [+0.0272, +0.0339] | 0.7617 | -0.1142 | -0.1007 | - |
| convnext_tiny_s43 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.5469 | 0.5239 | **+0.0230** | [+0.0210, +0.0250] | [+0.0212, +0.0248] | 0.7173 | -0.1704 | -0.1273 | - |
| convnext_tiny_s43 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.6424 | 0.6167 | **+0.0257** | [+0.0232, +0.0283] | [+0.0231, +0.0285] | 0.7801 | -0.1377 | -0.1128 | - |
| dinov2_vitb14 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.4169 | 0.4029 | **+0.0141** | [+0.0122, +0.0159] | [+0.0130, +0.0151] | 0.5992 | -0.1823 | -0.0661 | below_chance |
| dinov2_vitb14 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.3090 | 0.2940 | **+0.0149** | [+0.0132, +0.0167] | [+0.0140, +0.0158] | 0.5123 | -0.2033 | -0.0745 | below_chance |
| rad_dino | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9703 | 0.9377 | **+0.0326** | [+0.0272, +0.0381] | [+0.0273, +0.0383] | 0.9738 | -0.0034 | +0.0112 | - |
| rad_dino | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9277 | 0.8411 | **+0.0866** | [+0.0762, +0.0971] | [+0.0771, +0.0961] | 0.9299 | -0.0021 | +0.0255 | - |

## Delta_fit and A-gap — shenzhen_normal (descriptive, Shenzhen normal subset)

| run | scorer | d | K | n_groups_fit (fold 0 / 1) | ID_seen / ID_unseen | leaky | F2 | **Delta_fit** | jackknife 95% CI | bootstrap 95% CI (ref) | truth | A_gap | A_gap_cm | flags |
|---|---|---:|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| resnet50_s42 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.2050 | 0.1967 | **+0.0083** | [+0.0068, +0.0098] | [+0.0069, +0.0098] | 0.4392 | -0.2342 | -0.1266 | below_chance |
| resnet50_s42 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.825 / 0.813 | 0.2358 | 0.2217 | **+0.0141** | [+0.0121, +0.0160] | [+0.0122, +0.0159] | 0.4707 | -0.2349 | -0.1287 | below_chance |
| resnet50_s43 | mahalanobis_l2 | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.2369 | 0.2248 | **+0.0122** | [+0.0105, +0.0139] | [+0.0105, +0.0139] | 0.4824 | -0.2455 | -0.1302 | below_chance |
| resnet50_s43 | knn_mean_cosine | 2048 | 2 | 19,180 / 19,080 | 0.826 / 0.816 | 0.2868 | 0.2680 | **+0.0188** | [+0.0167, +0.0210] | [+0.0169, +0.0208] | 0.5263 | -0.2395 | -0.1312 | below_chance |
| convnext_tiny_s42 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.3551 | 0.3250 | **+0.0300** | [+0.0277, +0.0324] | [+0.0276, +0.0325] | 0.5360 | -0.1809 | -0.1289 | below_chance |
| convnext_tiny_s42 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.3480 | 0.3153 | **+0.0327** | [+0.0297, +0.0357] | [+0.0296, +0.0361] | 0.5091 | -0.1610 | -0.1226 | below_chance |
| convnext_tiny_s43 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.3062 | 0.2839 | **+0.0223** | [+0.0203, +0.0242] | [+0.0203, +0.0243] | 0.4968 | -0.1906 | -0.1325 | below_chance |
| convnext_tiny_s43 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.843 / 0.836 | 0.3341 | 0.3078 | **+0.0263** | [+0.0240, +0.0287] | [+0.0240, +0.0288] | 0.5177 | -0.1835 | -0.1340 | below_chance |
| dinov2_vitb14 | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.6189 | 0.6020 | **+0.0170** | [+0.0155, +0.0184] | [+0.0160, +0.0180] | 0.7785 | -0.1596 | -0.0508 | - |
| dinov2_vitb14 | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.766 / 0.741 | 0.5147 | 0.4987 | **+0.0160** | [+0.0133, +0.0186] | [+0.0150, +0.0170] | 0.7289 | -0.2143 | -0.0670 | - |
| rad_dino | mahalanobis_l2 | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9941 | 0.9843 | **+0.0098** | [+0.0074, +0.0122] | [+0.0074, +0.0122] | 0.9925 | +0.0016 | +0.0035 | ceiling |
| rad_dino | knn_mean_cosine | 768 | 2 | 19,180 / 19,080 | 0.833 / 0.807 | 0.9555 | 0.8635 | **+0.0920** | [+0.0825, +0.1014] | [+0.0815, +0.1025] | 0.9576 | -0.0022 | +0.0189 | - |

## Backbone level (primary OOD = Shenzhen)

CNN backbone estimate = mean of the two seeds' Delta_fit; robust = every seed's jackknife CI > 0.

| backbone | scorer | Delta_fit | robust | excluded (near-chance / ceiling / below-chance) | preregistered gate | post hoc gate |
|---|---|---:|---|---|---|---|
| resnet50 | mahalanobis_l2 | +0.0149 | yes | yes | fail | pass |
| resnet50 | knn_mean_cosine | +0.0186 | yes | yes | fail | pass |
| convnext_tiny | mahalanobis_l2 | +0.0262 | yes | yes | fail | pass |
| convnext_tiny | knn_mean_cosine | +0.0288 | yes | yes | fail | pass |
| dinov2_vitb14 | mahalanobis_l2 | +0.0155 | yes | no | pass | pass |
| dinov2_vitb14 | knn_mean_cosine | +0.0154 | yes | yes | pass | pass |
| rad_dino | mahalanobis_l2 | +0.0214 | yes | yes | pass | pass |
| rad_dino | knn_mean_cosine | +0.0893 | yes | no | pass | pass |

| verdict | mahalanobis_l2 | knn_mean_cosine |
|---|---|---|
| preregistered gate | INCONCLUSIVE (2 gate-passing backbone(s) < 3) | INCONCLUSIVE (2 gate-passing backbone(s) < 3) |
| post hoc (AUROC-only gate) | leakage not shown (0/4 gate-passing backbones with Delta_fit > 0.02; not robust: 0 with jackknife CI > 0) | leakage not shown (1/4 gate-passing backbones with Delta_fit > 0.02; not robust: 1 with jackknife CI > 0) |

Predictions (hold / fail; none gates anything):

| prediction | preregistered gate | post hoc gate |
|---|---|---|
| p1 | fails (0/2) | fails (0/4) |
| p3 | holds (1/2) | fails (1/4) |
| p2_mahalanobis_l2 | n/a (needs gate-passing FMs and CNNs) | holds (FM median +0.0184 vs CNN median +0.0205) |
| p2_mahalanobis_l2 (sensitivity, post hoc: excluded cells removed) | n/a (FMs left 1, CNNs left 0) | n/a (FMs left 1, CNNs left 0) |
| p2_knn_mean_cosine | n/a (needs gate-passing FMs and CNNs) | fails (FM median +0.0524 vs CNN median +0.0237) |
| p2_knn_mean_cosine (sensitivity, post hoc: excluded cells removed) | n/a (FMs left 1, CNNs left 0) | n/a (FMs left 1, CNNs left 0) |

## Case mix of ID_seen vs ID_unseen (A_gap_report)

| | seen | unseen |
|---|---|---|
| n | 16,497 | 4,045 |
| patients | 6,990 | 3,814 |
| AP share | 0.467 | 0.086 |
| No Finding | 0.497 | 0.717 |
| findings/image (mean) | 0.806 | 0.370 |
| >=2 findings | 0.211 | 0.067 |
| age median [IQR] | 49 [34, 59] | 47 [33, 58] |
| female | 0.431 | 0.469 |
| follow-up # (median) | 4.000 | 0.000 |
| images/patient in dataset (median) | 6.000 | 1.000 |
| Atelectasis | 0.114 | 0.051 |
| Cardiomegaly | 0.024 | 0.027 |
| Effusion | 0.138 | 0.033 |
| Infiltration | 0.193 | 0.114 |
| Mass | 0.056 | 0.032 |
| Nodule | 0.061 | 0.041 |
| Pneumonia | 0.015 | 0.003 |
| Pneumothorax | 0.058 | 0.005 |
| Consolidation | 0.048 | 0.013 |
| Edema | 0.025 | 0.002 |
| Emphysema | 0.028 | 0.007 |
| Fibrosis | 0.013 | 0.018 |
| Pleural_Thickening | 0.032 | 0.021 |
| Hernia | 0.002 | 0.003 |

Post-stratification: 19 of 32 strata kept; dropped share seen 0.060, unseen 0.007; effective sample size of the reweighted ID_seen 8,369 (of 15,503).

## Preprocessing parity (technical check before the report, 2026-10-08; scripts/r3/item8_p2a_parity_audit.py)

| source | n audited | format / PIL mode | bit depth | size w x h | median aspect w/h |
|---|---|---|---|---|---|
| nih | 1,000 | PNG L x994, PNG RGBA x6 | 8-bit | 1024-1024 x 1024-1024 (median 1024 x 1024) | 1.00 |
| shenzhen | 662 | PNG P x635, PNG RGB x27 | 8-bit | 1130-3001 x 948-3001 (median 2744 x 2937) | 1.00 |
| kermany | 5,856 | JPEG L x5,573, JPEG RGB x283 | 8-bit | 384-2916 x 127-2713 (median 1281 x 888) | 1.42 |

Every audited file decodes to uint8 (no 16-bit images, so no clipping by PIL convert('L')); the 635 Shenzhen palette PNGs have grey palettes (R = G = B for every used index); RGB / RGBA files are converted with PIL's ITU-R 601 luma.

| backbone | NIH chain | OOD chain |
|---|---|---|
| ResNet-50, ConvNeXt-T | original -> L -> 256 x 256 PIL bicubic (staged PNG) -> RGB -> Resize(256) + CenterCrop(224), torchvision bilinear -> ImageNet mean/std | same code path, same transform object |
| DINOv2-B | same 256 px staged PNG -> Resize(224, bicubic) + CenterCrop(224) -> ImageNet mean/std | same |
| RAD-DINO | original 1024 px -> L -> 518 x 518 PIL bicubic -> RGB -> /255 -> mean 0.5307 / std 0.2583 | same function on the original PNG / JPEG |

Mean grey level / fraction of pixels at 0 after staging: nih 129.6 / 0.023, shenzhen 156.7 / 0.006, kermany 122.8 / 0.047 (raw: nih 129.5 / 0.024, shenzhen 156.6 / 0.009, kermany 122.8 / 0.049); staging changes neither. Verdict: no source-dependent difference in the chain, so no OOD re-extraction or rescoring. The between-source intensity difference is already present in the raw files.

## Deviations and caveats

1. **Competence gate (post hoc sensitivity, approved 2026-10-08):** the preregistered CNN condition 'final train loss <= 50% of epoch-1 loss' was copied from single-label cross-entropy training (medbench) and is mis-specified for 14-finding multi-label BCE, where the epoch-1 mean loss is already low. The preregistered verdict applies it as written; the post hoc verdict uses the AUROC condition only, with the precommitted thresholds (CNN macro AUROC >= 0.75, FM probe >= 0.70); no new thresholds.
2. Wang-test images of the checkpoint patients (test_ckpt) are neither seen nor unseen and are excluded (counted above).
3. Preprocessing parity (see the section above): within each backbone NIH and OOD share one code path. RAD-DINO does not use its BitImageProcessor (shortest edge 518 + centre crop 518); it squashes to 518 x 518, identical for square NIH images, not for non-square OOD images (Kermany pediatric median aspect w/h 1.42). The ceiling pilot read DINOv2-B inputs from the 1024 px originals.
4. RAD-DINO loaded with transformers 4.44.2 (private install; the environment's transformers needs torch >= 2.5), pooler_output (CLS after the final layer norm). RAD-DINO pretraining saw all NIH images (seen and unseen alike).
5. CNN backbone-level aggregation over the two seeds (mean; robust = both seeds' jackknife CIs > 0) is not specified in the precommit and was fixed before the cells were computed. Cells excluded by G_near_chance / G_ceiling / below-chance do not meet the bar; the denominator stays the number of gate-passing backbones (precommit wording).
6. No logit scores (precommit): CXR reports the scorer channel (A-fit) and the case-mix-corrected A-gap for feature scores only; no family comparison.
7. Bootstrap CIs are reference only (fitted scores held fixed; too narrow when Delta > 0 per the coverage study); the jackknife decides robustness.
8. **A_gap is descriptive.** It is not evidence that the image-level split raises or lowers AUROC: DINOv2-B never saw NIH and still shows A_gap_cm -0.059 (mahalanobis_l2) / -0.071 (knn_mean_cosine) on Shenzhen after post-stratification on view x finding pattern, so residual case-mix confounding between ID_seen and ID_unseen remains.

