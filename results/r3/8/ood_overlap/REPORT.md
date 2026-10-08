# R3 item 8 (post hoc): OOD / ID patient overlap and OOD-restricted sensitivity

Commit: 10398d7

Verdict: Track A Kermany, OOD restricted to patients absent from every ID set: median Δ change -0.0014 (range -0.0059 to -0.0002); status vs 0.02 changes in 2 of 24 rows.

Post hoc: requested after the P2-b Kermany tie finding; not in any precommit. Precommitted estimands unchanged.

## Overlap counts (ID sets vs OOD)

| arm | group unit | OOD images (groups) | OOD groups in each ID set | OOD groups in any ID set | OOD images from those groups | clean OOD images (groups) | pixel-identical ID-OOD pairs per ID set | cross-group identical pairs |
|---|---|---|---|---|---|---|---|---|
| kermany_b0 | patient ID | 8866 (882) | train 219, val 158, seen 176, unseen 166 | 387 | 5170 | 3696 (495) | train 116, val 9, seen 19, unseen 109 | 0 |
| kermany_b1 | patient ID | 8866 (882) | train 165, val 128, seen 137, unseen 221 | 387 | 5170 | 3696 (495) | train 89, val 10, seen 10, unseen 144 | 0 |
| kermany_std | patient ID | 8866 (716) | train 387, val 261, seen 73, unseen 0, unseen_extra 0 | 398 | 5512 | 3354 (318) | train 181, val 23, seen 0, unseen 0, unseen_extra 0 | 0 |
| isic2019_b0 | lesion_id, else image (no patient ID in ISIC 2019) | 492 (248) | train 0, val 0, seen 0, unseen 0 | 0 | 0 | 492 (248) | train 0, val 0, seen 0, unseen 0 | 0 |
| isic2019_b1 | lesion_id, else image (no patient ID in ISIC 2019) | 492 (248) | train 0, val 0, seen 0, unseen 0 | 0 | 0 | 492 (248) | train 0, val 0, seen 0, unseen 0 | 0 |
| isic2019_std | lesion_id, else image (no patient ID in ISIC 2019) | 492 (248) | train 0, val 0, seen 0, unseen 0 | 0 | 0 | 492 (248) | train 0, val 0, seen 0, unseen 0 | 0 |

## Track A Kermany (std arm): full OOD vs clean OOD

Same cells, folds, scorers and settings as item 1 v2 (seen = strict); only OOD rows are removed. n_groups_fit = n_groups_train / K per fold.

| cell | scorer | n_groups_fit | K | d | ID acc | n OOD full / clean | AUROC seen full / clean | AUROC unseen full / clean | Δ full | Δ clean | old CI full | old CI clean | new CI full | new CI clean | status 0.02 full / clean | status 0 full / clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kermany_convnext_tiny_s42_std | mahalanobis_l2 | 4240/2 | 2 | 768 | 0.9985 | 8866 / 3354 | 0.6480 / 0.6786 | 0.6203 / 0.6513 | 0.0278 | 0.0273 | [0.0217, 0.0344] | [0.0212, 0.0345] | [0.0214, 0.0342] | [0.0204, 0.0343] | pass_both / pass_both | pass_both / pass_both |
| kermany_convnext_tiny_s42_std | knn_mean_cosine | 4240/2 | 2 | 768 | 0.9985 | 8866 / 3354 | 0.7398 / 0.7697 | 0.7040 / 0.7358 | 0.0359 | 0.0339 | [0.0276, 0.0443] | [0.0256, 0.0427] | [0.0275, 0.0442] | [0.0254, 0.0423] | pass_both / pass_both | pass_both / pass_both |
| kermany_convnext_tiny_s43_std | mahalanobis_l2 | 4240/2 | 2 | 768 | 0.9985 | 8866 / 3354 | 0.5867 / 0.6130 | 0.5576 / 0.5841 | 0.0291 | 0.0289 | [0.0232, 0.0353] | [0.0230, 0.0353] | [0.0232, 0.0350] | [0.0228, 0.0350] | pass_both / pass_both | pass_both / pass_both |
| kermany_convnext_tiny_s43_std | knn_mean_cosine | 4240/2 | 2 | 768 | 0.9985 | 8866 / 3354 | 0.6934 / 0.7181 | 0.6500 / 0.6765 | 0.0434 | 0.0416 | [0.0355, 0.0511] | [0.0338, 0.0496] | [0.0356, 0.0511] | [0.0335, 0.0498] | pass_both / pass_both | pass_both / pass_both |
| kermany_densenet121_s42_std | mahalanobis_l2 | 4240/2 | 2 | 1024 | 0.9940 | 8866 / 3354 | 0.7581 / 0.7872 | 0.7294 / 0.7599 | 0.0287 | 0.0273 | [0.0231, 0.0344] | [0.0221, 0.0326] | [0.0225, 0.0349] | [0.0215, 0.0331] | pass_both / pass_both | pass_both / pass_both |
| kermany_densenet121_s42_std | knn_mean_cosine | 4240/2 | 2 | 1024 | 0.9940 | 8866 / 3354 | 0.7876 / 0.8075 | 0.7559 / 0.7772 | 0.0317 | 0.0303 | [0.0247, 0.0385] | [0.0238, 0.0369] | [0.0242, 0.0391] | [0.0233, 0.0373] | pass_both / pass_both | pass_both / pass_both |
| kermany_densenet121_s43_std | mahalanobis_l2 | 4240/2 | 2 | 1024 | 0.9955 | 8866 / 3354 | 0.8047 / 0.8281 | 0.7801 / 0.8045 | 0.0246 | 0.0236 | [0.0205, 0.0290] | [0.0196, 0.0277] | [0.0201, 0.0291] | [0.0194, 0.0278] | pass_both / fail_both | pass_both / pass_both |
| kermany_densenet121_s43_std | knn_mean_cosine | 4240/2 | 2 | 1024 | 0.9955 | 8866 / 3354 | 0.8316 / 0.8533 | 0.8019 / 0.8256 | 0.0297 | 0.0277 | [0.0242, 0.0350] | [0.0227, 0.0325] | [0.0228, 0.0366] | [0.0213, 0.0341] | pass_both / pass_both | pass_both / pass_both |
| kermany_effb3_s42_std | mahalanobis_l2 | 4240/2 | 2 | 1536 | 0.9940 | 8866 / 3354 | 0.7537 / 0.7794 | 0.7333 / 0.7598 | 0.0205 | 0.0196 | [0.0172, 0.0235] | [0.0165, 0.0226] | [0.0171, 0.0238] | [0.0164, 0.0228] | fail_both / fail_both | pass_both / pass_both |
| kermany_effb3_s42_std | knn_mean_cosine | 4240/2 | 2 | 1536 | 0.9940 | 8866 / 3354 | 0.7954 / 0.8182 | 0.7743 / 0.7983 | 0.0211 | 0.0199 | [0.0168, 0.0256] | [0.0158, 0.0240] | [0.0165, 0.0256] | [0.0153, 0.0244] | fail_both / fail_both | pass_both / pass_both |
| kermany_efficientnet_v2_s_s42_std | mahalanobis_l2 | 4240/2 | 2 | 1280 | 0.9970 | 8866 / 3354 | 0.8044 / 0.8244 | 0.7811 / 0.8022 | 0.0232 | 0.0222 | [0.0189, 0.0275] | [0.0179, 0.0266] | [0.0191, 0.0273] | [0.0181, 0.0263] | fail_both / fail_both | pass_both / pass_both |
| kermany_efficientnet_v2_s_s42_std | knn_mean_cosine | 4240/2 | 2 | 1280 | 0.9970 | 8866 / 3354 | 0.8446 / 0.8619 | 0.8259 / 0.8446 | 0.0187 | 0.0174 | [0.0149, 0.0229] | [0.0136, 0.0214] | [0.0145, 0.0229] | [0.0130, 0.0217] | fail_both / fail_both | pass_both / pass_both |
| kermany_mobilenet_v3_large_s42_std | mahalanobis_l2 | 4240/2 | 2 | 960 | 0.9940 | 8866 / 3354 | 0.7925 / 0.8242 | 0.7717 / 0.8041 | 0.0208 | 0.0201 | [0.0171, 0.0245] | [0.0164, 0.0238] | [0.0175, 0.0241] | [0.0166, 0.0236] | fail_both / fail_both | pass_both / pass_both |
| kermany_mobilenet_v3_large_s42_std | knn_mean_cosine | 4240/2 | 2 | 960 | 0.9940 | 8866 / 3354 | 0.8376 / 0.8658 | 0.8083 / 0.8398 | 0.0293 | 0.0261 | [0.0234, 0.0355] | [0.0210, 0.0313] | [0.0234, 0.0351] | [0.0208, 0.0313] | pass_both / pass_both | pass_both / pass_both |
| kermany_regnet_y_3_2gf_s42_std | mahalanobis_l2 | 4240/2 | 2 | 1512 | 0.9955 | 8866 / 3354 | 0.7657 / 0.7954 | 0.7126 / 0.7452 | 0.0532 | 0.0501 | [0.0452, 0.0613] | [0.0424, 0.0585] | [0.0435, 0.0628] | [0.0408, 0.0595] | pass_both / pass_both | pass_both / pass_both |
| kermany_regnet_y_3_2gf_s42_std | knn_mean_cosine | 4240/2 | 2 | 1512 | 0.9955 | 8866 / 3354 | 0.8392 / 0.8618 | 0.7764 / 0.8049 | 0.0628 | 0.0569 | [0.0501, 0.0759] | [0.0450, 0.0688] | [0.0488, 0.0769] | [0.0435, 0.0703] | pass_both / pass_both | pass_both / pass_both |
| kermany_resnet18_s42_std | mahalanobis_l2 | 4240/2 | 2 | 512 | 0.9970 | 8866 / 3354 | 0.8208 / 0.8446 | 0.8042 / 0.8291 | 0.0166 | 0.0155 | [0.0131, 0.0199] | [0.0123, 0.0188] | [0.0124, 0.0208] | [0.0114, 0.0196] | fail_both / fail_both | pass_both / pass_both |
| kermany_resnet18_s42_std | knn_mean_cosine | 4240/2 | 2 | 512 | 0.9970 | 8866 / 3354 | 0.8702 / 0.8889 | 0.8507 / 0.8708 | 0.0195 | 0.0181 | [0.0151, 0.0238] | [0.0137, 0.0222] | [0.0139, 0.0252] | [0.0124, 0.0237] | fail_both / fail_both | pass_both / pass_both |
| kermany_resnet18_s43_std | mahalanobis_l2 | 4240/2 | 2 | 512 | 0.9985 | 8866 / 3354 | 0.7650 / 0.7839 | 0.7407 / 0.7602 | 0.0243 | 0.0237 | [0.0196, 0.0288] | [0.0192, 0.0281] | [0.0193, 0.0293] | [0.0188, 0.0286] | fail_both / fail_both | pass_both / pass_both |
| kermany_resnet18_s43_std | knn_mean_cosine | 4240/2 | 2 | 512 | 0.9985 | 8866 / 3354 | 0.8344 / 0.8503 | 0.8082 / 0.8251 | 0.0263 | 0.0252 | [0.0212, 0.0315] | [0.0200, 0.0307] | [0.0189, 0.0336] | [0.0173, 0.0332] | pass_old_only / fail_both | pass_both / pass_both |
| kermany_resnet50_s42_std | mahalanobis_l2 | 4240/2 | 2 | 2048 | 0.9940 | 8866 / 3354 | 0.8010 / 0.8229 | 0.7564 / 0.7804 | 0.0446 | 0.0425 | [0.0385, 0.0510] | [0.0363, 0.0484] | [0.0377, 0.0516] | [0.0358, 0.0492] | pass_both / pass_both | pass_both / pass_both |
| kermany_resnet50_s42_std | knn_mean_cosine | 4240/2 | 2 | 2048 | 0.9940 | 8866 / 3354 | 0.8184 / 0.8489 | 0.7980 / 0.8300 | 0.0205 | 0.0189 | [0.0154, 0.0253] | [0.0142, 0.0236] | [0.0138, 0.0271] | [0.0125, 0.0253] | fail_both / fail_both | pass_both / pass_both |
| kermany_resnet50_s43_std | mahalanobis_l2 | 4240/2 | 2 | 2048 | 0.9955 | 8866 / 3354 | 0.8138 / 0.8377 | 0.7722 / 0.7975 | 0.0416 | 0.0402 | [0.0360, 0.0473] | [0.0346, 0.0461] | [0.0367, 0.0465] | [0.0351, 0.0453] | pass_both / pass_both | pass_both / pass_both |
| kermany_resnet50_s43_std | knn_mean_cosine | 4240/2 | 2 | 2048 | 0.9955 | 8866 / 3354 | 0.8284 / 0.8539 | 0.8102 / 0.8376 | 0.0182 | 0.0162 | [0.0146, 0.0221] | [0.0130, 0.0195] | [0.0138, 0.0227] | [0.0122, 0.0203] | fail_both / fail_both | pass_both / pass_both |

## Caveats

1. ISIC 2019 has no patient ID; overlap is checked at lesion and pixel level only. No overlap was found, so no ISIC sensitivity run.
