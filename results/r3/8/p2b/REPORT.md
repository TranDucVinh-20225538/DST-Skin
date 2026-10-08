# R3 item 8 / P2-b: scorer and model channels on the medbench (B) arm

Commit: 6731516

Verdict: Kermany p1 (scorer channel > 0) holds, p2 (model channel > 0) holds; identification identified.

Precommit: results/r3/8/PRECOMMIT.json, P2_b_retrain_second_medical_dataset. Cached medbench (B) features, CPU, crossfit_ood Track A scorers; jackknife over patients / lesion groups.

## Predictions

- p1 (Kermany scorer channel > 0, both scorers): holds (robust: holds): mahalanobis_l2 median +0.0119 (median lower bound +0.0082, n = 16); knn_mean_cosine median +0.0109 (median lower bound +0.0060, n = 16)
- p2 (Kermany model channel > 0, both scorers): holds (robust: fails): mahalanobis_l2 median +0.0133 (median lower bound -0.0051, n = 16); knn_mean_cosine median +0.0147 (median lower bound -0.0033, n = 16)
- p3 (ISIC 2019 same direction): fails (ISIC scorer channel > 0 for both scorers vs Kermany > 0; model channel ISIC not > 0 vs Kermany > 0)
  - ISIC scorer channel: holds (robust: holds): mahalanobis_l2 median +0.0430 (median lower bound +0.0371, n = 16); knn_mean_cosine median +0.0269 (median lower bound +0.0219, n = 16)
  - ISIC model channel: fails (robust: fails): mahalanobis_l2 median -0.0418 (median lower bound -0.0645, n = 16); knn_mean_cosine median -0.0334 (median lower bound -0.0548, n = 16)

## Identification check

- Kermany: identified (per-fold median differences, mahalanobis_l2: scorer 0.0044, model 0.0066; knn_mean_cosine: scorer 0.0012, model 0.0125; threshold 0.02)
- ISIC 2019: identified (per-fold median differences, mahalanobis_l2: scorer 0.0012, model 0.0046; knn_mean_cosine: scorer 0.0008, model 0.0143; threshold 0.02)

## Secondary: fraction of the gap closed

- Kermany OCT, mahalanobis_l2: median +0.5063 over 16 cells with |leaky - truth| >= 0.01 (0 of 16 counted cells excluded)
- Kermany OCT, knn_mean_cosine: median +0.4602 over 16 cells with |leaky - truth| >= 0.01 (0 of 16 counted cells excluded)
- ISIC 2019, mahalanobis_l2: median -1.6584 over 9 cells with |leaky - truth| >= 0.01 (7 of 16 counted cells excluded)
- ISIC 2019, knn_mean_cosine: median -1.4819 over 7 cells with |leaky - truth| >= 0.01 (9 of 16 counted cells excluded)

## Cells

No earlier CI exists for these channels (new quantities); the jackknife CI is the only CI. n_groups_fit = training patients / groups of the fold; K = 2 (within-S folds).

| dataset | arch | seed | fold | scorer | leaky | truth | F2 | scorer channel [CI] | model channel [CI] | fraction closed | n_groups_fit | K | d | ID acc seen / unseen | quality | flags | orphan seen dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kermany OCT | resnet18 | 42 | b0 | mahalanobis_l2 | 0.7201 | 0.6953 | 0.7133 | +0.0068 [+0.0025, +0.0111] | +0.0180 [-0.0032, +0.0392] | +0.2746 | 2232 | 2 | 512 | 0.9916 / 0.9802 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet18 | 42 | b0 | knn_mean_cosine | 0.7635 | 0.7316 | 0.7549 | +0.0086 [+0.0015, +0.0157] | +0.0234 [+0.0046, +0.0421] | +0.2681 | 2232 | 2 | 512 | 0.9916 / 0.9802 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet18 | 42 | b1 | mahalanobis_l2 | 0.7787 | 0.7607 | 0.7689 | +0.0098 [+0.0059, +0.0137] | +0.0082 [-0.0097, +0.0261] | +0.5451 | 2214 | 2 | 512 | 0.9893 / 0.9837 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet18 | 42 | b1 | knn_mean_cosine | 0.8305 | 0.8128 | 0.8206 | +0.0099 [+0.0061, +0.0137] | +0.0078 [-0.0066, +0.0221] | +0.5603 | 2214 | 2 | 512 | 0.9893 / 0.9837 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet18 | 43 | b0 | mahalanobis_l2 | 0.7202 | 0.6963 | 0.7140 | +0.0061 [+0.0024, +0.0098] | +0.0178 [-0.0054, +0.0410] | +0.2559 | 2232 | 2 | 512 | 0.9913 / 0.9785 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet18 | 43 | b0 | knn_mean_cosine | 0.7468 | 0.7181 | 0.7405 | +0.0063 [-0.0004, +0.0130] | +0.0224 [+0.0017, +0.0431] | +0.2193 | 2232 | 2 | 512 | 0.9913 / 0.9785 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet18 | 43 | b1 | mahalanobis_l2 | 0.7539 | 0.7384 | 0.7433 | +0.0106 [+0.0068, +0.0144] | +0.0049 [-0.0146, +0.0245] | +0.6822 | 2214 | 2 | 512 | 0.9879 / 0.9813 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet18 | 43 | b1 | knn_mean_cosine | 0.8100 | 0.7940 | 0.7999 | +0.0100 [+0.0060, +0.0141] | +0.0060 [-0.0094, +0.0213] | +0.6269 | 2214 | 2 | 512 | 0.9879 / 0.9813 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet50 | 42 | b0 | mahalanobis_l2 | 0.6788 | 0.6528 | 0.6671 | +0.0117 [+0.0082, +0.0152] | +0.0144 [-0.0032, +0.0320] | +0.4482 | 2232 | 2 | 2048 | 0.9931 / 0.9821 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet50 | 42 | b0 | knn_mean_cosine | 0.7228 | 0.7002 | 0.7198 | +0.0030 [-0.0027, +0.0086] | +0.0196 [+0.0024, +0.0367] | +0.1319 | 2232 | 2 | 2048 | 0.9931 / 0.9821 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet50 | 42 | b1 | mahalanobis_l2 | 0.7584 | 0.7310 | 0.7432 | +0.0153 [+0.0128, +0.0177] | +0.0122 [-0.0040, +0.0284] | +0.5562 | 2214 | 2 | 2048 | 0.9891 / 0.9827 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet50 | 42 | b1 | knn_mean_cosine | 0.7913 | 0.7765 | 0.7851 | +0.0062 [+0.0036, +0.0087] | +0.0086 [-0.0057, +0.0228] | +0.4180 | 2214 | 2 | 2048 | 0.9891 / 0.9827 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet50 | 43 | b0 | mahalanobis_l2 | 0.7143 | 0.6852 | 0.7007 | +0.0136 [+0.0101, +0.0171] | +0.0155 [-0.0051, +0.0361] | +0.4676 | 2232 | 2 | 2048 | 0.9906 / 0.9800 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet50 | 43 | b0 | knn_mean_cosine | 0.7398 | 0.7159 | 0.7375 | +0.0023 [-0.0026, +0.0072] | +0.0215 [+0.0029, +0.0401] | +0.0966 | 2232 | 2 | 2048 | 0.9906 / 0.9800 | pass | - | 2 (1 groups) |
| Kermany OCT | resnet50 | 43 | b1 | mahalanobis_l2 | 0.7393 | 0.7095 | 0.7213 | +0.0180 [+0.0147, +0.0212] | +0.0118 [-0.0051, +0.0287] | +0.6030 | 2214 | 2 | 2048 | 0.9897 / 0.9824 | pass | - | 3 (2 groups) |
| Kermany OCT | resnet50 | 43 | b1 | knn_mean_cosine | 0.7857 | 0.7723 | 0.7784 | +0.0072 [+0.0039, +0.0106] | +0.0061 [-0.0097, +0.0219] | +0.5415 | 2214 | 2 | 2048 | 0.9897 / 0.9824 | pass | - | 3 (2 groups) |
| Kermany OCT | densenet121 | 42 | b0 | mahalanobis_l2 | 0.7129 | 0.6867 | 0.7064 | +0.0066 [+0.0008, +0.0123] | +0.0196 [+0.0004, +0.0388] | +0.2505 | 2232 | 2 | 1024 | 0.9932 / 0.9818 | pass | - | 2 (1 groups) |
| Kermany OCT | densenet121 | 42 | b0 | knn_mean_cosine | 0.7126 | 0.6766 | 0.6960 | +0.0166 [+0.0068, +0.0264] | +0.0194 [-0.0012, +0.0400] | +0.4611 | 2232 | 2 | 1024 | 0.9932 / 0.9818 | pass | - | 2 (1 groups) |
| Kermany OCT | densenet121 | 42 | b1 | mahalanobis_l2 | 0.7693 | 0.7351 | 0.7572 | +0.0121 [+0.0083, +0.0159] | +0.0221 [+0.0039, +0.0402] | +0.3535 | 2214 | 2 | 1024 | 0.9901 / 0.9829 | pass | - | 3 (2 groups) |
| Kermany OCT | densenet121 | 42 | b1 | knn_mean_cosine | 0.8115 | 0.7805 | 0.7927 | +0.0188 [+0.0137, +0.0239] | +0.0122 [-0.0024, +0.0269] | +0.6053 | 2214 | 2 | 1024 | 0.9901 / 0.9829 | pass | - | 3 (2 groups) |
| Kermany OCT | densenet121 | 43 | b0 | mahalanobis_l2 | 0.7143 | 0.6896 | 0.7087 | +0.0056 [+0.0007, +0.0106] | +0.0191 [-0.0011, +0.0393] | +0.2281 | 2232 | 2 | 1024 | 0.9931 / 0.9811 | pass | - | 2 (1 groups) |
| Kermany OCT | densenet121 | 43 | b0 | knn_mean_cosine | 0.7381 | 0.7070 | 0.7263 | +0.0118 [+0.0058, +0.0177] | +0.0193 [+0.0008, +0.0378] | +0.3792 | 2232 | 2 | 1024 | 0.9931 / 0.9811 | pass | - | 2 (1 groups) |
| Kermany OCT | densenet121 | 43 | b1 | mahalanobis_l2 | 0.7897 | 0.7604 | 0.7788 | +0.0109 [+0.0073, +0.0144] | +0.0185 [+0.0010, +0.0360] | +0.3703 | 2214 | 2 | 1024 | 0.9909 / 0.9829 | pass | - | 3 (2 groups) |
| Kermany OCT | densenet121 | 43 | b1 | knn_mean_cosine | 0.8346 | 0.8068 | 0.8219 | +0.0127 [+0.0094, +0.0159] | +0.0151 [-0.0001, +0.0304] | +0.4556 | 2214 | 2 | 1024 | 0.9909 / 0.9829 | pass | - | 3 (2 groups) |
| Kermany OCT | convnext_tiny | 42 | b0 | mahalanobis_l2 | 0.6741 | 0.6493 | 0.6577 | +0.0163 [+0.0117, +0.0209] | +0.0084 [-0.0178, +0.0346] | +0.6606 | 2232 | 2 | 768 | 0.9939 / 0.9803 | pass | - | 2 (1 groups) |
| Kermany OCT | convnext_tiny | 42 | b0 | knn_mean_cosine | 0.7366 | 0.7064 | 0.7207 | +0.0159 [+0.0094, +0.0223] | +0.0143 [-0.0094, +0.0380] | +0.5258 | 2232 | 2 | 768 | 0.9939 / 0.9803 | pass | - | 2 (1 groups) |
| Kermany OCT | convnext_tiny | 42 | b1 | mahalanobis_l2 | 0.7287 | 0.7098 | 0.7118 | +0.0168 [+0.0129, +0.0208] | +0.0020 [-0.0252, +0.0293] | +0.8923 | 2214 | 2 | 768 | 0.9894 / 0.9833 | pass | - | 3 (2 groups) |
| Kermany OCT | convnext_tiny | 42 | b1 | knn_mean_cosine | 0.7742 | 0.7510 | 0.7524 | +0.0217 [+0.0160, +0.0275] | +0.0014 [-0.0237, +0.0266] | +0.9377 | 2214 | 2 | 768 | 0.9894 / 0.9833 | pass | - | 3 (2 groups) |
| Kermany OCT | convnext_tiny | 43 | b0 | mahalanobis_l2 | 0.6482 | 0.6229 | 0.6329 | +0.0153 [+0.0110, +0.0196] | +0.0101 [-0.0157, +0.0359] | +0.6030 | 2232 | 2 | 768 | 0.9929 / 0.9837 | pass | - | 2 (1 groups) |
| Kermany OCT | convnext_tiny | 43 | b0 | knn_mean_cosine | 0.7087 | 0.6761 | 0.6937 | +0.0150 [+0.0081, +0.0218] | +0.0176 [-0.0043, +0.0395] | +0.4593 | 2232 | 2 | 768 | 0.9929 / 0.9837 | pass | - | 2 (1 groups) |
| Kermany OCT | convnext_tiny | 43 | b1 | mahalanobis_l2 | 0.7599 | 0.7390 | 0.7400 | +0.0199 [+0.0162, +0.0237] | +0.0010 [-0.0247, +0.0267] | +0.9520 | 2214 | 2 | 768 | 0.9899 / 0.9815 | pass | - | 3 (2 groups) |
| Kermany OCT | convnext_tiny | 43 | b1 | knn_mean_cosine | 0.7981 | 0.7763 | 0.7747 | +0.0233 [+0.0179, +0.0288] | -0.0016 [-0.0247, +0.0215] | +1.0741 | 2214 | 2 | 768 | 0.9899 / 0.9815 | pass | - | 3 (2 groups) |
| ISIC 2019 | resnet18 | 42 | b0 | mahalanobis_l2 | 0.7356 | 0.7593 | 0.6964 | +0.0392 [+0.0337, +0.0447] | -0.0629 [-0.0859, -0.0398] | -1.6584 | 6315 | 2 | 512 | 0.7812 / 0.7310 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet18 | 42 | b0 | knn_mean_cosine | 0.7305 | 0.7467 | 0.7011 | +0.0293 [+0.0236, +0.0351] | -0.0456 [-0.0684, -0.0227] | -1.8088 | 6315 | 2 | 512 | 0.7812 / 0.7310 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet18 | 42 | b1 | mahalanobis_l2 | 0.7075 | 0.7277 | 0.6679 | +0.0395 [+0.0344, +0.0447] | -0.0598 [-0.0804, -0.0391] | -1.9531 | 6395 | 2 | 512 | 0.7853 / 0.7223 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet18 | 42 | b1 | knn_mean_cosine | 0.7064 | 0.7317 | 0.6752 | +0.0312 [+0.0260, +0.0364] | -0.0565 [-0.0764, -0.0365] | -1.2338 | 6395 | 2 | 512 | 0.7853 / 0.7223 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet18 | 43 | b0 | mahalanobis_l2 | 0.7134 | 0.7317 | 0.6721 | +0.0413 [+0.0347, +0.0479] | -0.0596 [-0.0842, -0.0351] | -2.2570 | 6315 | 2 | 512 | 0.7755 / 0.7380 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet18 | 43 | b0 | knn_mean_cosine | 0.7280 | 0.7413 | 0.6947 | +0.0333 [+0.0269, +0.0397] | -0.0466 [-0.0685, -0.0247] | -2.5147 | 6315 | 2 | 512 | 0.7755 / 0.7380 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet18 | 43 | b1 | mahalanobis_l2 | 0.6974 | 0.7177 | 0.6571 | +0.0403 [+0.0344, +0.0462] | -0.0607 [-0.0824, -0.0390] | -1.9813 | 6395 | 2 | 512 | 0.7778 / 0.7213 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet18 | 43 | b1 | knn_mean_cosine | 0.6869 | 0.7071 | 0.6570 | +0.0299 [+0.0248, +0.0349] | -0.0500 [-0.0706, -0.0295] | -1.4819 | 6395 | 2 | 512 | 0.7778 / 0.7213 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet50 | 42 | b0 | mahalanobis_l2 | 0.7896 | 0.7568 | 0.7275 | +0.0621 [+0.0536, +0.0707] | -0.0293 [-0.0507, -0.0079] | +1.8924 | 6315 | 2 | 2048 | 0.8018 / 0.7382 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet50 | 42 | b0 | knn_mean_cosine | 0.7480 | 0.7504 | 0.7272 | +0.0207 [+0.0155, +0.0259] | -0.0231 [-0.0443, -0.0020] | n/a | 6315 | 2 | 2048 | 0.8018 / 0.7382 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet50 | 42 | b1 | mahalanobis_l2 | 0.8033 | 0.7729 | 0.7422 | +0.0611 [+0.0533, +0.0689] | -0.0307 [-0.0514, -0.0099] | +2.0077 | 6395 | 2 | 2048 | 0.8229 / 0.7279 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet50 | 42 | b1 | knn_mean_cosine | 0.7414 | 0.7382 | 0.7191 | +0.0223 [+0.0179, +0.0267] | -0.0191 [-0.0381, -0.0001] | n/a | 6395 | 2 | 2048 | 0.8229 / 0.7279 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet50 | 43 | b0 | mahalanobis_l2 | 0.7889 | 0.7623 | 0.7207 | +0.0683 [+0.0588, +0.0777] | -0.0417 [-0.0663, -0.0170] | +2.5658 | 6315 | 2 | 2048 | 0.8035 / 0.7353 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet50 | 43 | b0 | knn_mean_cosine | 0.7404 | 0.7428 | 0.7137 | +0.0267 [+0.0204, +0.0331] | -0.0291 [-0.0515, -0.0067] | n/a | 6315 | 2 | 2048 | 0.8035 / 0.7353 | pass | - | 37 (33 groups) |
| ISIC 2019 | resnet50 | 43 | b1 | mahalanobis_l2 | 0.7775 | 0.7514 | 0.7100 | +0.0675 [+0.0591, +0.0759] | -0.0414 [-0.0611, -0.0216] | +2.5844 | 6395 | 2 | 2048 | 0.8296 / 0.7320 | pass | - | 32 (32 groups) |
| ISIC 2019 | resnet50 | 43 | b1 | knn_mean_cosine | 0.7109 | 0.7102 | 0.6884 | +0.0225 [+0.0190, +0.0260] | -0.0218 [-0.0399, -0.0037] | n/a | 6395 | 2 | 2048 | 0.8296 / 0.7320 | pass | - | 32 (32 groups) |
| ISIC 2019 | densenet121 | 42 | b0 | mahalanobis_l2 | 0.7506 | 0.7435 | 0.7120 | +0.0386 [+0.0319, +0.0452] | -0.0315 [-0.0512, -0.0118] | n/a | 6315 | 2 | 1024 | 0.8026 / 0.7408 | pass | - | 37 (33 groups) |
| ISIC 2019 | densenet121 | 42 | b0 | knn_mean_cosine | 0.7520 | 0.7594 | 0.7235 | +0.0284 [+0.0228, +0.0341] | -0.0359 [-0.0573, -0.0144] | n/a | 6315 | 2 | 1024 | 0.8026 / 0.7408 | pass | - | 37 (33 groups) |
| ISIC 2019 | densenet121 | 42 | b1 | mahalanobis_l2 | 0.7328 | 0.7245 | 0.6934 | +0.0393 [+0.0331, +0.0456] | -0.0310 [-0.0508, -0.0112] | n/a | 6395 | 2 | 1024 | 0.8104 / 0.7352 | pass | - | 32 (32 groups) |
| ISIC 2019 | densenet121 | 42 | b1 | knn_mean_cosine | 0.7186 | 0.7180 | 0.6906 | +0.0280 [+0.0224, +0.0336] | -0.0274 [-0.0476, -0.0072] | n/a | 6395 | 2 | 1024 | 0.8104 / 0.7352 | pass | - | 32 (32 groups) |
| ISIC 2019 | densenet121 | 43 | b0 | mahalanobis_l2 | 0.7565 | 0.7491 | 0.7163 | +0.0402 [+0.0335, +0.0470] | -0.0328 [-0.0535, -0.0121] | n/a | 6315 | 2 | 1024 | 0.8150 / 0.7404 | pass | - | 37 (33 groups) |
| ISIC 2019 | densenet121 | 43 | b0 | knn_mean_cosine | 0.7374 | 0.7419 | 0.7110 | +0.0265 [+0.0215, +0.0314] | -0.0310 [-0.0524, -0.0096] | n/a | 6315 | 2 | 1024 | 0.8150 / 0.7404 | pass | - | 37 (33 groups) |
| ISIC 2019 | densenet121 | 43 | b1 | mahalanobis_l2 | 0.7392 | 0.7375 | 0.6955 | +0.0437 [+0.0371, +0.0503] | -0.0420 [-0.0612, -0.0229] | n/a | 6395 | 2 | 1024 | 0.8112 / 0.7304 | pass | - | 32 (32 groups) |
| ISIC 2019 | densenet121 | 43 | b1 | knn_mean_cosine | 0.7130 | 0.7232 | 0.6815 | +0.0316 [+0.0260, +0.0371] | -0.0418 [-0.0610, -0.0226] | -3.0922 | 6395 | 2 | 1024 | 0.8112 / 0.7304 | pass | - | 32 (32 groups) |
| ISIC 2019 | convnext_tiny | 42 | b0 | mahalanobis_l2 | 0.7440 | 0.7475 | 0.6965 | +0.0475 [+0.0420, +0.0530] | -0.0510 [-0.0764, -0.0256] | n/a | 6315 | 2 | 768 | 0.8569 / 0.7523 | pass | - | 37 (33 groups) |
| ISIC 2019 | convnext_tiny | 42 | b0 | knn_mean_cosine | 0.7074 | 0.7270 | 0.6803 | +0.0271 [+0.0232, +0.0309] | -0.0467 [-0.0701, -0.0233] | -1.3818 | 6315 | 2 | 768 | 0.8569 / 0.7523 | pass | - | 37 (33 groups) |
| ISIC 2019 | convnext_tiny | 42 | b1 | mahalanobis_l2 | 0.7430 | 0.7407 | 0.7006 | +0.0424 [+0.0371, +0.0476] | -0.0401 [-0.0627, -0.0174] | n/a | 6395 | 2 | 768 | 0.8805 / 0.7408 | pass | - | 32 (32 groups) |
| ISIC 2019 | convnext_tiny | 42 | b1 | knn_mean_cosine | 0.7147 | 0.7145 | 0.6929 | +0.0218 [+0.0183, +0.0253] | -0.0215 [-0.0429, -0.0002] | n/a | 6395 | 2 | 768 | 0.8805 / 0.7408 | pass | - | 32 (32 groups) |
| ISIC 2019 | convnext_tiny | 43 | b0 | mahalanobis_l2 | 0.6724 | 0.6846 | 0.6254 | +0.0470 [+0.0413, +0.0527] | -0.0593 [-0.0826, -0.0359] | -3.8331 | 6315 | 2 | 768 | 0.8676 / 0.7522 | pass | - | 37 (33 groups) |
| ISIC 2019 | convnext_tiny | 43 | b0 | knn_mean_cosine | 0.6534 | 0.6717 | 0.6294 | +0.0240 [+0.0205, +0.0275] | -0.0424 [-0.0639, -0.0209] | -1.3070 | 6315 | 2 | 768 | 0.8676 / 0.7522 | pass | - | 37 (33 groups) |
| ISIC 2019 | convnext_tiny | 43 | b1 | mahalanobis_l2 | 0.7257 | 0.7214 | 0.6749 | +0.0509 [+0.0448, +0.0569] | -0.0465 [-0.0697, -0.0233] | n/a | 6395 | 2 | 768 | 0.8764 / 0.7454 | pass | - | 32 (32 groups) |
| ISIC 2019 | convnext_tiny | 43 | b1 | knn_mean_cosine | 0.7123 | 0.7103 | 0.6880 | +0.0243 [+0.0210, +0.0277] | -0.0223 [-0.0439, -0.0008] | n/a | 6395 | 2 | 768 | 0.8764 / 0.7454 | pass | - | 32 (32 groups) |

## Post hoc sensitivity: OOD restricted to patients absent from every ID set (Kermany)

Requested after the overlap audit (results/r3/8/ood_overlap/audit.json); not in the precommit. ISIC 2019 shows no lesion or pixel overlap, so it has no sensitivity run.

| arm | OOD images | from patients in an ID set | clean OOD images (patients) | seen-OOD pixel-identical pairs | train-OOD pairs |
|---|---|---|---|---|---|
| Kermany b0 | 8866 | 5170 | 3696 (495) | 19 | 116 |
| Kermany b1 | 8866 | 5170 | 3696 (495) | 10 | 89 |

| scorer | statistic | median, all OOD (n cells) | median, clean OOD (n cells) | median lower bound, all | median lower bound, clean |
|---|---|---|---|---|---|
| mahalanobis_l2 | leaky | +0.7244 (16) | +0.7592 (16) | +0.7029 | +0.7371 |
| mahalanobis_l2 | F2 | +0.7137 (16) | +0.7475 (16) | +0.6916 | +0.7251 |
| mahalanobis_l2 | truth | +0.7029 (16) | +0.7357 (16) | +0.6803 | +0.7134 |
| mahalanobis_l2 | scorer_channel | +0.0119 (16) | +0.0112 (16) | +0.0082 | +0.0078 |
| mahalanobis_l2 | model_channel | +0.0133 (16) | +0.0127 (16) | -0.0051 | -0.0049 |
| knn_mean_cosine | leaky | +0.7688 (16) | +0.7990 (16) | +0.7424 | +0.7720 |
| knn_mean_cosine | F2 | +0.7537 (16) | +0.7858 (16) | +0.7271 | +0.7589 |
| knn_mean_cosine | truth | +0.7413 (16) | +0.7729 (16) | +0.7160 | +0.7465 |
| knn_mean_cosine | scorer_channel | +0.0109 (16) | +0.0096 (16) | +0.0060 | +0.0047 |
| knn_mean_cosine | model_channel | +0.0147 (16) | +0.0150 (16) | -0.0033 | -0.0030 |

## Deviations and caveats

1. Technical deviation: the package check (leaky and F2 vs crossfit_auroc) allows 1e-9 plus one pair per near-tied seen / OOD score pair. Kermany b1 kNN failed the plain 1e-9 check by exactly one pair (7.5647508e-09) because pixel-identical seen / OOD images give tied scores that the two code paths break differently. Estimates unchanged; package values and tie counts are stored per cell. Cells computed before the fix passed the plain check.
2. Dataset finding: Kermany OCT has pixel-identical images across classes of the same patient; seen-OOD identical pairs: 19 (b0), 10 (b1).
3. Kermany OOD (DRUSEN) shares patients with the ID sets (table above); the precommitted estimand is kept, the restricted-OOD rerun is post hoc.
4. Orphan seen images (group with no training image after the medbench val carve-out) are dropped, as in item 1; counts per cell.
5. ISIC 2019 has no patient ID; groups are lesions (or single images), so patient-level overlap cannot be checked.
