# R3 item 5b: pooling check and fold-count stability

commit: 843fecf

Post hoc relative to the paper's original precommit; criteria fixed in PRECOMMIT.json (fe65a1e) before Step B ran. Camelyon17, 30 training slides (3 hospitals x 10), 5 FMs, Track A scorers.

## Step A: pooling check (cached Track A scores, no refits)

Verdict: the Mahalanobis gap persists under matched pooling in 2 of 3 failing cells (dinov2_vitb14, dinov2_vitl14); matched pooling reduces it.

Rule (precommit): naive pooling explains the gap if, in each failing Mahalanobis cell at seed 42 (DINOv2-B, DINOv2-L, CONCH), |F2_matched - F1| <= 0.02 and |F2_naive - F1| > 0.02.

| FM | seed | scorer | leaky | F1 | F2_naive | F2_matched | F2_naive - F1 | F2_matched - F1 | n_id fold 0 / 1 | s.d. OOD fit 0 / fit 1 / averaged | corr(OOD fit 0, fit 1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uni | 42 | Mahalanobis | 0.9988 | 0.9918 | 0.9928 | 0.9927 | +0.0010 | +0.0009 | 11021 / 22539 | 4.341 / 4.328 / 3.952 | 0.662 |
| uni | 42 | kNN | 0.9967 | 0.9029 | 0.8881 | 0.8942 | -0.0148 | -0.0087 | 11021 / 22539 | 0.08007 / 0.06096 / 0.06594 | 0.744 |
| uni | 43 | Mahalanobis | 0.9988 | 0.9960 | 0.9930 | 0.9959 | -0.0030 | -0.0001 | 16225 / 17335 | 4.43 / 3.805 / 4.03 | 0.916 |
| uni | 43 | kNN | 0.9967 | 0.8836 | 0.8750 | 0.8803 | -0.0086 | -0.0032 | 16225 / 17335 | 0.06699 / 0.06412 / 0.06376 | 0.892 |
| uni | 44 | Mahalanobis | 0.9988 | 0.9949 | 0.9953 | 0.9952 | +0.0004 | +0.0003 | 20577 / 12983 | 4.15 / 3.951 / 3.937 | 0.889 |
| uni | 44 | kNN | 0.9967 | 0.8922 | 0.8812 | 0.8809 | -0.0109 | -0.0112 | 20577 / 12983 | 0.06606 / 0.05933 / 0.06135 | 0.915 |
| virchow2 | 42 | Mahalanobis | 0.9857 | 0.9584 | 0.9664 | 0.9659 | +0.0079 | +0.0075 | 11021 / 22539 | 18.18 / 24.6 / 20.67 | 0.865 |
| virchow2 | 42 | kNN | 0.9636 | 0.8583 | 0.8542 | 0.8632 | -0.0041 | +0.0049 | 11021 / 22539 | 0.07262 / 0.08571 / 0.07625 | 0.854 |
| virchow2 | 43 | Mahalanobis | 0.9857 | 0.9646 | 0.9546 | 0.9641 | -0.0100 | -0.0006 | 16225 / 17335 | 22.64 / 20.44 / 21.36 | 0.966 |
| virchow2 | 43 | kNN | 0.9636 | 0.8136 | 0.8056 | 0.8102 | -0.0080 | -0.0034 | 16225 / 17335 | 0.08177 / 0.08023 / 0.07993 | 0.948 |
| virchow2 | 44 | Mahalanobis | 0.9857 | 0.9667 | 0.9699 | 0.9681 | +0.0032 | +0.0014 | 20577 / 12983 | 23.13 / 19.04 / 20.92 | 0.969 |
| virchow2 | 44 | kNN | 0.9636 | 0.8451 | 0.8368 | 0.8368 | -0.0084 | -0.0083 | 20577 / 12983 | 0.08321 / 0.0788 / 0.08037 | 0.969 |
| dinov2_vitb14 | 42 | Mahalanobis | 0.7297 | 0.6549 | 0.7036 | 0.6796 | +0.0487 | +0.0248 | 11021 / 22539 | 10.47 / 14.51 / 12.47 | 0.993 |
| dinov2_vitb14 | 42 | kNN | 0.7556 | 0.6296 | 0.6355 | 0.6446 | +0.0059 | +0.0150 | 11021 / 22539 | 0.04432 / 0.04569 / 0.04491 | 0.992 |
| dinov2_vitb14 | 43 | Mahalanobis | 0.7297 | 0.6607 | 0.6617 | 0.6607 | +0.0011 | -0.0000 | 16225 / 17335 | 12.43 / 13.14 / 12.77 | 0.996 |
| dinov2_vitb14 | 43 | kNN | 0.7556 | 0.6336 | 0.6332 | 0.6329 | -0.0004 | -0.0007 | 16225 / 17335 | 0.04504 / 0.04534 / 0.04511 | 0.993 |
| dinov2_vitb14 | 44 | Mahalanobis | 0.7297 | 0.6748 | 0.6975 | 0.6823 | +0.0226 | +0.0075 | 20577 / 12983 | 14.11 / 11.38 / 12.72 | 0.991 |
| dinov2_vitb14 | 44 | kNN | 0.7556 | 0.6499 | 0.6518 | 0.6516 | +0.0019 | +0.0017 | 20577 / 12983 | 0.04582 / 0.04527 / 0.04543 | 0.990 |
| dinov2_vitl14 | 42 | Mahalanobis | 0.7609 | 0.6826 | 0.7342 | 0.7057 | +0.0516 | +0.0231 | 11021 / 22539 | 16.06 / 23.86 / 19.93 | 0.994 |
| dinov2_vitl14 | 42 | kNN | 0.7940 | 0.6718 | 0.6786 | 0.6872 | +0.0068 | +0.0153 | 11021 / 22539 | 0.0436 / 0.04531 / 0.04441 | 0.996 |
| dinov2_vitl14 | 43 | Mahalanobis | 0.7609 | 0.6934 | 0.6932 | 0.6932 | -0.0002 | -0.0002 | 16225 / 17335 | 20.16 / 20.4 / 20.27 | 0.997 |
| dinov2_vitl14 | 43 | kNN | 0.7940 | 0.6765 | 0.6761 | 0.6756 | -0.0003 | -0.0008 | 16225 / 17335 | 0.04459 / 0.04447 / 0.04449 | 0.997 |
| dinov2_vitl14 | 44 | Mahalanobis | 0.7609 | 0.7042 | 0.7259 | 0.7131 | +0.0216 | +0.0088 | 20577 / 12983 | 22.75 / 17.62 / 20.17 | 0.996 |
| dinov2_vitl14 | 44 | kNN | 0.7940 | 0.6962 | 0.6971 | 0.7002 | +0.0009 | +0.0039 | 20577 / 12983 | 0.04529 / 0.04397 / 0.04458 | 0.996 |
| conch_v1_5 | 42 | Mahalanobis | 0.8575 | 0.7696 | 0.8033 | 0.7843 | +0.0337 | +0.0147 | 11021 / 22539 | 4.379 / 6.558 / 5.377 | 0.931 |
| conch_v1_5 | 42 | kNN | 0.9609 | 0.8124 | 0.7924 | 0.8029 | -0.0200 | -0.0094 | 11021 / 22539 | 0.03083 / 0.02949 / 0.02888 | 0.833 |
| conch_v1_5 | 43 | Mahalanobis | 0.8575 | 0.7714 | 0.7651 | 0.7692 | -0.0063 | -0.0023 | 16225 / 17335 | 5.759 / 5.374 / 5.539 | 0.980 |
| conch_v1_5 | 43 | kNN | 0.9609 | 0.7722 | 0.7681 | 0.7678 | -0.0042 | -0.0045 | 16225 / 17335 | 0.02921 / 0.0291 / 0.02845 | 0.905 |
| conch_v1_5 | 44 | Mahalanobis | 0.8575 | 0.7866 | 0.7961 | 0.7869 | +0.0095 | +0.0003 | 20577 / 12983 | 6.125 / 4.94 / 5.496 | 0.973 |
| conch_v1_5 | 44 | kNN | 0.9609 | 0.8148 | 0.8005 | 0.8004 | -0.0144 | -0.0144 | 20577 / 12983 | 0.02896 / 0.02847 / 0.02827 | 0.939 |

F2_matched uses the same two arms as F1 (ID of fold k vs OOD, both scored by the fit that did not see fold k); it differs from F1 only in the arm weights (n_k vs 1/2), so F2_matched - F1 = sum_k (n_k / N - 1/2) AUROC_k. It is non-zero where the fold sizes are unequal and the two arm AUROCs differ (arm AUROCs in stepA.csv). n_groups_fit = 15 slides per fit, K = 2; d and probe ID accuracy as in the Step B table.

## Step B: fold-count stability (K = 2, 5, 10; folds stratified within hospital)

Verdict: not stable: |X_5 - X_10| <= 0.01 on Mahalanobis 3/5 evaluated, kNN 2/5 evaluated; recommend the largest affordable K and report the drift.

X_K = matched-pooled cross-fit AUROC (crossfit_ood, protocol default, fold_ids = slide fold of each id_val patch). Leaky = scorer fitted on all 30 slides. Criterion: |X_5 - X_10| <= 0.01 on >= 4/5 FMs for both scorers; |X_2 - X_5| is reported only.

| FM | scorer | d | probe ID acc | leaky | X_2 | X_5 | X_10 | abs(X_5 - X_10) | abs(X_2 - X_5) | n_groups_fit K=2/5/10 | fit patches (min-max) K=2/5/10 | 95% CI X_5 / X_10 (jackknife) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uni | Mahalanobis | 1024 | 0.994 | 0.9988 | 0.9946 | 0.9949 | 0.9946 | 0.0003 | 0.0003 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| uni | kNN | 1024 | 0.994 | 0.9967 | 0.8544 | 0.8992 | 0.9037 | 0.0046 | 0.0448 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| virchow2 | Mahalanobis | 2560 | 0.995 | 0.9857 | 0.9643 | 0.9646 | 0.9639 | 0.0007 | 0.0003 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| virchow2 | kNN | 2560 | 0.995 | 0.9636 | 0.8288 | 0.8395 | 0.8424 | 0.0029 | 0.0106 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| dinov2_vitb14 | Mahalanobis | 768 | 0.972 | 0.7297 | 0.6536 | 0.6590 | 0.6745 | 0.0155 | 0.0054 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| dinov2_vitb14 | kNN | 768 | 0.972 | 0.7556 | 0.6181 | 0.6379 | 0.6582 | 0.0204 | 0.0197 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| dinov2_vitl14 | Mahalanobis | 1024 | 0.976 | 0.7609 | 0.6860 | 0.6889 | 0.7038 | 0.0149 | 0.0030 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| dinov2_vitl14 | kNN | 1024 | 0.976 | 0.7940 | 0.6632 | 0.6744 | 0.6955 | 0.0211 | 0.0112 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| conch_v1_5 | Mahalanobis | 768 | 0.987 | 0.8575 | 0.7619 | 0.7673 | 0.7722 | 0.0049 | 0.0054 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |
| conch_v1_5 | kNN | 768 | 0.987 | 0.9609 | 0.7742 | 0.7913 | 0.8027 | 0.0114 | 0.0171 | 15-15 / 24-24 / 27-27 | 141361-161075 / 196779-264241 / 234080-289101 | point only / point only |

### Post hoc: leave-one-slide-out (K = 30) and the deployment bracket (FMs)

| model | scorer | d | probe ID acc | leaky | X_2 | X_5 | X_10 | X_30 | leaky - X_2 | leaky - X_10 | leaky - X_30 | n_groups_fit K=2/5/10/30 | ceiling (> 0.98 under leaky and every X_K) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uni | Mahalanobis | 1024 | 0.994 | 0.9988 | 0.9946 | 0.9949 | 0.9946 | 0.9942 | +0.0042 | +0.0042 | +0.0046 | 15 / 24 / 27 / 29 | CEILING |
| uni | kNN | 1024 | 0.994 | 0.9967 | 0.8544 | 0.8992 | 0.9037 | 0.9026 | +0.1423 | +0.0930 | +0.0941 | 15 / 24 / 27 / 29 |  |
| virchow2 | Mahalanobis | 2560 | 0.995 | 0.9857 | 0.9643 | 0.9646 | 0.9639 | 0.9620 | +0.0214 | +0.0219 | +0.0237 | 15 / 24 / 27 / 29 |  |
| virchow2 | kNN | 2560 | 0.995 | 0.9636 | 0.8288 | 0.8395 | 0.8424 | 0.8402 | +0.1347 | +0.1212 | +0.1234 | 15 / 24 / 27 / 29 |  |
| dinov2_vitb14 | Mahalanobis | 768 | 0.972 | 0.7297 | 0.6536 | 0.6590 | 0.6745 | 0.6720 | +0.0761 | +0.0552 | +0.0578 | 15 / 24 / 27 / 29 |  |
| dinov2_vitb14 | kNN | 768 | 0.972 | 0.7556 | 0.6181 | 0.6379 | 0.6582 | 0.6572 | +0.1375 | +0.0974 | +0.0984 | 15 / 24 / 27 / 29 |  |
| dinov2_vitl14 | Mahalanobis | 1024 | 0.976 | 0.7609 | 0.6860 | 0.6889 | 0.7038 | 0.7012 | +0.0750 | +0.0571 | +0.0598 | 15 / 24 / 27 / 29 |  |
| dinov2_vitl14 | kNN | 1024 | 0.976 | 0.7940 | 0.6632 | 0.6744 | 0.6955 | 0.6943 | +0.1308 | +0.0985 | +0.0998 | 15 / 24 / 27 / 29 |  |
| conch_v1_5 | Mahalanobis | 768 | 0.987 | 0.8575 | 0.7619 | 0.7673 | 0.7722 | 0.7677 | +0.0956 | +0.0853 | +0.0898 | 15 / 24 / 27 / 29 |  |
| conch_v1_5 | kNN | 768 | 0.987 | 0.9609 | 0.7742 | 0.7913 | 0.8027 | 0.8004 | +0.1867 | +0.1582 | +0.1606 | 15 / 24 / 27 / 29 |  |

Ceiling (rule of the CXR precommit, results/r3/8/PRECOMMIT.json G_ceiling): AUROC > 0.98 under both variants, i.e. leaky and every computed X_K; leaky - X_K is then bounded by 1 - X_K and small by construction, not evidence of absent leakage.
Near ceiling (descriptive, no flag): Virchow2 Mahalanobis, leaky 0.986 / X_2 0.964; the gap has little room above X_K.
Drift (post hoc, observed): X_10 > X_2 in 9/10 rows; X_30 < X_10 in 10/10 rows (X_30 - X_10 from -0.0044 to -0.0004). X_K does not keep rising beyond K = 10, so the ordering leaky - X_30 <= leaky - X_10 assumed for a deployment bracket does not hold; leaky - X_K is reported as the observed drift across fit sizes, not as a bound on deployment inflation. leaky - X_K uses the default protocol (same ID patches, different fits), not the Track A paper_2fold Delta_fit.

## Post hoc: Camelyon CNNs (seed 42), same estimator, K = 2 / 5 / 10 / 30

Not in PRECOMMIT.json; requested after Step B to bracket the main Delta_fit numbers between K = 2 (preregistered fit size, 15 slides) and K = 10 / 30 (closer to the deployed fit size, 30 slides). kNN on the GPU scorer after an equality check against the CPU Step B result (conch_v1_5, K = 2: max abs difference 0.0, Delta identical). Virchow2 kNN K = 10 (FM table) also uses the GPU scorer: the CPU run timed out at 7 h.

| model | scorer | d | model ID acc | leaky | X_2 | X_5 | X_10 | X_30 | leaky - X_2 | leaky - X_10 | leaky - X_30 | n_groups_fit K=2/5/10/30 | ceiling (> 0.98 under leaky and every X_K) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | Mahalanobis | 2048 | 0.995 | 0.9178 | 0.6555 | 0.6464 | 0.7545 | 0.7492 | +0.2623 | +0.1632 | +0.1685 | 15 / 24 / 27 / 29 |  |
| resnet50 | kNN | 2048 | 0.995 | 0.8664 | 0.6253 | 0.6161 | 0.7058 | 0.7008 | +0.2411 | +0.1606 | +0.1656 | 15 / 24 / 27 / 29 |  |
| convnext_tiny | Mahalanobis | 768 | 0.997 | 0.9283 | 0.8741 | 0.8727 | 0.8710 | 0.8687 | +0.0542 | +0.0573 | +0.0596 | 15 / 24 / 27 / 29 |  |
| convnext_tiny | kNN | 768 | 0.997 | 0.9154 | 0.8639 | 0.8604 | 0.8570 | 0.8549 | +0.0515 | +0.0584 | +0.0605 | 15 / 24 / 27 / 29 |  |
| densenet121 | Mahalanobis | 1024 | 0.996 | 0.9884 | 0.8561 | 0.8535 | 0.9299 | 0.9279 | +0.1322 | +0.0585 | +0.0605 | 15 / 24 / 27 / 29 |  |
| densenet121 | kNN | 1024 | 0.996 | 0.9735 | 0.8682 | 0.8666 | 0.9014 | 0.8999 | +0.1053 | +0.0721 | +0.0736 | 15 / 24 / 27 / 29 |  |
| effb3 | Mahalanobis | 1536 | 0.996 | 0.9767 | 0.8485 | 0.8342 | 0.8870 | 0.8847 | +0.1282 | +0.0897 | +0.0920 | 15 / 24 / 27 / 29 |  |
| effb3 | kNN | 1536 | 0.996 | 0.9338 | 0.7643 | 0.7585 | 0.8122 | 0.8108 | +0.1695 | +0.1216 | +0.1230 | 15 / 24 / 27 / 29 |  |
| efficientnet_v2_s | Mahalanobis | 1280 | 0.996 | 0.8799 | 0.8159 | 0.8072 | 0.8216 | 0.8200 | +0.0640 | +0.0583 | +0.0598 | 15 / 24 / 27 / 29 |  |
| efficientnet_v2_s | kNN | 1280 | 0.996 | 0.8558 | 0.7736 | 0.7747 | 0.7885 | 0.7875 | +0.0821 | +0.0673 | +0.0682 | 15 / 24 / 27 / 29 |  |
| mobilenet_v3_large | Mahalanobis | 960 | 0.997 | 0.8694 | 0.7881 | 0.7882 | 0.7993 | 0.7967 | +0.0813 | +0.0701 | +0.0727 | 15 / 24 / 27 / 29 |  |
| mobilenet_v3_large | kNN | 960 | 0.997 | 0.9038 | 0.8057 | 0.8159 | 0.8275 | 0.8265 | +0.0981 | +0.0763 | +0.0773 | 15 / 24 / 27 / 29 |  |
| regnet_y_3_2gf | Mahalanobis | 1512 | 0.996 | 0.9642 | 0.8270 | 0.8274 | 0.8536 | 0.8498 | +0.1372 | +0.1105 | +0.1144 | 15 / 24 / 27 / 29 |  |
| regnet_y_3_2gf | kNN | 1512 | 0.996 | 0.9561 | 0.8269 | 0.8248 | 0.8462 | 0.8440 | +0.1292 | +0.1099 | +0.1121 | 15 / 24 / 27 / 29 |  |
| resnet18 | Mahalanobis | 512 | 0.995 | 0.9611 | 0.7709 | 0.7453 | 0.8126 | 0.8100 | +0.1902 | +0.1485 | +0.1511 | 15 / 24 / 27 / 29 |  |
| resnet18 | kNN | 512 | 0.995 | 0.9354 | 0.7775 | 0.7584 | 0.7908 | 0.7885 | +0.1580 | +0.1446 | +0.1469 | 15 / 24 / 27 / 29 |  |

Ceiling (rule of the CXR precommit, results/r3/8/PRECOMMIT.json G_ceiling): AUROC > 0.98 under both variants, i.e. leaky and every computed X_K; leaky - X_K is then bounded by 1 - X_K and small by construction, not evidence of absent leakage.
Drift (post hoc, observed): X_10 > X_2 in 14/16 rows; X_30 < X_10 in 16/16 rows (X_30 - X_10 from -0.0053 to -0.0010). X_K does not keep rising beyond K = 10, so the ordering leaky - X_30 <= leaky - X_10 assumed for a deployment bracket does not hold; leaky - X_K is reported as the observed drift across fit sizes, not as a bound on deployment inflation. leaky - X_K uses the default protocol (same ID patches, different fits), not the Track A paper_2fold Delta_fit.

## Decision (precommit table)

- Step A: gap reported as unexplained; recommend group-disjoint evaluation for Mahalanobis.
- Step B: recommend the largest affordable K and report the drift.

