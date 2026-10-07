# Medbench group-leakage check: summary

Precommit `decisions/precommit_group_leakage_medbench_2026-10-06.md` (+ deviations file). Gap = AUROC(OOD vs
seen-group ID) − AUROC(OOD vs unseen-group ID), class-matched; Δ_fit = same-group − group-disjoint 2-fold
scorer-fit AUROC; positive = inflation. Per arch = mean over gate-passing seeds (BreakHis: and repeats).
`*` = every run's 95% cluster-bootstrap CI excludes 0; `u` = underpowered cell (excluded from bars).

## dermamnist

(A) frozen M_std:

| arch | gap MSP | gap Energy | gap ELogitNorm | gap ViM | gap ReAct | gap Mahalanobis | gap kNN | Δfit Mahalanobis | Δfit kNN | Δfit ViM | Δfit ReAct | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | -0.015 | -0.016 | -0.016 | -0.007 | -0.144* | -0.012 | -0.020* | +0.023* | +0.005* | +0.045* | -0.000 | 0.921 / 0.898 |
| densenet121 | -0.020 | +0.002 | +0.029* | -0.027 | -0.003 | -0.017* | -0.046* | +0.015* | +0.006* | +0.001 | +0.000 | 0.873 / 0.886 |
| effb3 | -0.029* | -0.005 | +0.001 | +0.022* | +0.009 | +0.005 | -0.022* | +0.023* | +0.008* | +0.028* | +0.000 | 0.882 / 0.898 |
| efficientnet_v2_s | +0.016 | +0.016 | +0.007 | +0.013 | +0.023* | +0.009 | -0.005 | +0.014* | +0.004* | +0.035* | +0.000 | 0.899 / 0.886 |
| mobilenet_v3_large | -0.058* | -0.021* | +0.062* | -0.023* | -0.010 | +0.001 | -0.066* | +0.028* | +0.013* | +0.024* | +0.000 | 0.827 / 0.883 |
| regnet_y_3_2gf | -0.062* | -0.012 | -0.004 | -0.008 | -0.009 | -0.027* | -0.046* | +0.029* | +0.011* | +0.039* | +0.000 | 0.867 / 0.887 |
| resnet18 | -0.053* | -0.022 | -0.030 | -0.010 | -0.007 | -0.029* | -0.060* | +0.025* | +0.008* | +0.034* | +0.000 | 0.839 / 0.880 |
| resnet50 | -0.039* | -0.020 | -0.025 | +0.012 | -0.007 | -0.006 | -0.035* | +0.019* | +0.006* | +0.037* | +0.000 | 0.859 / 0.894 |

(B) retrained group-disjoint, within-model gap (mean over seeds × folds):

| arch | MSP | Energy | ELogitNorm | ViM | ReAct | Mahalanobis | kNN | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | +0.030 | +0.029 | +0.019 | +0.034* | -0.055 | +0.030* | +0.021 | 0.906 / 0.823 |
| densenet121 | -0.003 | +0.016 | +0.048 | -0.016 | -0.003 | +0.008 | -0.001 | 0.855 / 0.812 |
| resnet18 | -0.017 | +0.010 | +0.019 | +0.008 | +0.005 | +0.008 | -0.005 | 0.837 / 0.807 |
| resnet50 | +0.023 | +0.032 | +0.026 | +0.025 | +0.028 | +0.015 | +0.000 | 0.886 / 0.812 |

Ranking (A, 12 runs): Mahalanobis→Mahalanobis 11, Mahalanobis→ViM 1; median tau-b 0.81; CI-significant swaps 0
Ranking (B, 16 runs): Mahalanobis→Mahalanobis 16; median tau-b 0.90; CI-significant swaps 0

## isic2019

(A) frozen M_std:

| arch | gap MSP | gap Energy | gap ELogitNorm | gap ViM | gap ReAct | gap Mahalanobis | gap kNN | Δfit Mahalanobis | Δfit kNN | Δfit ViM | Δfit ReAct | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | -0.053* | -0.064* | -0.080* | -0.043* | -0.124* | -0.074* | -0.076* | +0.043* | +0.019* | +0.048* | -0.000 | 0.868 / 0.862 |
| densenet121 | -0.081* | -0.084* | -0.074* | -0.094* | -0.070* | -0.094* | -0.108* | +0.035* | +0.024* | +0.012 | -0.000 | 0.809 / 0.860 |
| effb3 | -0.078* | -0.077* | -0.084* | -0.034* | -0.072* | -0.057* | -0.087* | +0.046* | +0.017* | +0.039* | +0.000 | 0.833 / 0.864 |
| efficientnet_v2_s | -0.031* | -0.035* | -0.040* | -0.033* | -0.034* | -0.046* | -0.060* | +0.033* | +0.009* | +0.043* | +0.000 | 0.877 / 0.852 |
| mobilenet_v3_large | -0.090* | -0.092* | -0.086* | -0.060* | -0.086* | -0.080* | -0.110* | +0.036* | +0.020* | +0.024* | -0.000 | 0.751 / 0.843 |
| regnet_y_3_2gf | -0.085* | -0.088* | -0.089* | -0.040* | -0.078* | -0.086* | -0.089* | +0.067* | +0.030* | +0.035* | +0.000 | 0.801 / 0.852 |
| resnet18 | -0.105* | -0.114* | -0.103* | -0.070* | -0.100* | -0.091* | -0.096* | +0.029* | +0.020* | +0.016* | -0.000 | 0.778 / 0.852 |
| resnet50 | -0.083* | -0.099* | -0.105* | -0.061* | -0.092* | -0.073* | -0.099* | +0.057* | +0.016* | +0.045* | -0.000 | 0.804 / 0.852 |

(B) retrained group-disjoint, within-model gap (mean over seeds × folds):

| arch | MSP | Energy | ELogitNorm | ViM | ReAct | Mahalanobis | kNN | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | +0.056* | +0.045* | +0.017 | +0.015 | +0.017 | +0.031* | +0.023 | 0.870 / 0.748 |
| densenet121 | +0.017 | +0.014 | +0.003 | -0.014 | -0.007 | +0.008 | -0.000 | 0.810 / 0.737 |
| resnet18 | +0.006 | -0.008 | -0.020 | +0.004 | -0.013 | -0.001 | -0.005 | 0.780 / 0.728 |
| resnet50 | +0.033* | +0.021 | +0.002 | +0.039* | +0.013 | +0.042* | +0.009 | 0.814 / 0.733 |

Ranking (A, 12 runs): Mahalanobis→Mahalanobis 4, kNN→kNN 4, Mahalanobis→kNN 3, ViM→kNN 1; median tau-b 0.81; CI-significant swaps 1
Ranking (B, 16 runs): Mahalanobis→Mahalanobis 12, Mahalanobis→kNN 2, kNN→kNN 2; median tau-b 0.81; CI-significant swaps 0

## kermany

(A) frozen M_std:

| arch | gap MSP | gap Energy | gap ELogitNorm | gap ViM | gap ReAct | gap Mahalanobis | gap kNN | Δfit Mahalanobis | Δfit kNN | Δfit ViM | Δfit ReAct | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | +0.133* | +0.147* | +0.148* | +0.250* | +0.087* | +0.362* | +0.331* | +0.028* | +0.040* | +0.030 | -0.000 | 0.998 / 1.000 |
| densenet121 | +0.156* | +0.159* | +0.162* | +0.068* | +0.094* | +0.476* | +0.398* | +0.027* | +0.031* | +0.016 | +0.000 | 0.995 / 1.000 |
| effb3 | +0.168* | +0.190* | +0.201* | +0.222* | +0.162* | +0.312* | +0.304* | +0.020* | +0.021* | +0.020* | -0.000 | 0.994 / 1.000 |
| efficientnet_v2_s | +0.151* | +0.167* | +0.169* | +0.206* | +0.149* | +0.346* | +0.301* | +0.023* | +0.019* | +0.049* | -0.000 | 0.997 / 1.000 |
| mobilenet_v3_large | +0.238* | +0.252* | +0.260* | +0.256* | +0.230* | +0.440* | +0.398* | +0.021* | +0.029* | +0.030* | +0.000 | 0.994 / 0.988 |
| regnet_y_3_2gf | +0.159* | +0.148* | +0.150* | +0.207* | +0.157* | +0.491* | +0.477* | +0.053* | +0.063* | +0.032* | +0.000 | 0.995 / 1.000 |
| resnet18 | +0.213* | +0.214* | +0.197* | +0.240* | +0.184* | +0.414* | +0.339* | +0.020* | +0.023* | +0.014* | +0.000 | 0.998 / 1.000 |
| resnet50 | +0.164* | +0.175* | +0.158* | +0.274* | +0.151* | +0.599* | +0.486* | +0.043* | +0.019* | +0.039* | +0.000 | 0.995 / 0.994 |

(B) retrained group-disjoint, within-model gap (mean over seeds × folds):

| arch | MSP | Energy | ELogitNorm | ViM | ReAct | Mahalanobis | kNN | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | +0.010 | +0.009 | +0.008 | +0.021 | +0.007 | +0.024 | +0.028 | 0.992 / 0.982 |
| densenet121 | +0.016 | +0.016 | +0.016 | +0.014 | +0.013 | +0.029* | +0.032* | 0.992 / 0.982 |
| resnet18 | +0.014 | +0.014 | +0.012 | +0.016 | +0.012 | +0.021 | +0.024 | 0.990 / 0.981 |
| resnet50 | +0.012 | +0.011 | +0.010 | +0.024* | +0.010 | +0.029* | +0.019 | 0.991 / 0.982 |

Ranking (A, 12 runs): kNN→ELogitNorm 8, kNN→ReAct 2, kNN→Energy 1, kNN→MSP 1; median tau-b 0.10; CI-significant swaps 12
Ranking (B, 16 runs): kNN→kNN 15, ELogitNorm→ELogitNorm 1; median tau-b 0.90; CI-significant swaps 0

## breakhis

(A) frozen M_std:

| arch | gap MSP | gap Energy | gap ELogitNorm | gap ViM | gap ReAct | gap Mahalanobis | gap kNN | Δfit Mahalanobis | Δfit kNN | Δfit ViM | Δfit ReAct | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | +0.136 | +0.129 | +0.097 | +0.097 | +0.111 | +0.155 | +0.147 | +0.131* | +0.092* | +0.112* | +0.000 | 0.915 / 0.684 |
| densenet121 | +0.136 | +0.145 | +0.114 | +0.066 | +0.080 | +0.170* | +0.155* | +0.252* | +0.178* | +0.062* | +0.000 | 0.904 / 0.677 |
| effb3 | +0.161 | +0.161* | +0.116 | +0.130* | +0.136 | +0.184* | +0.162* | +0.175* | +0.121* | +0.151* | +0.000 | 0.925 / 0.678 |
| efficientnet_v2_s | +0.153 | +0.154 | +0.108 | +0.140* | +0.131 | +0.178* | +0.173* | +0.124* | +0.073* | +0.105* | -0.000 | 0.908 / 0.660 |
| mobilenet_v3_large | +0.105 | +0.120 | +0.096 | +0.091 | +0.094 | +0.148* | +0.141 | +0.179* | +0.147* | +0.098* | +0.000 | 0.883 / 0.653 |
| regnet_y_3_2gf | +0.127 | +0.119 | +0.088 | +0.096* | +0.092 | +0.168 | +0.153 | +0.262* | +0.214* | +0.123* | +0.000 | 0.904 / 0.673 |
| resnet18 | +0.129 | +0.137 | +0.112* | +0.113 | +0.122 | +0.149* | +0.148* | +0.208* | +0.162* | +0.110* | +0.000 | 0.886 / 0.651 |
| resnet50 | +0.147 | +0.153* | +0.129* | +0.104 | +0.138* | +0.193* | +0.157* | +0.264* | +0.173* | +0.160* | +0.000 | 0.905 / 0.676 |

Ranking (A, 60 runs): Mahalanobis→ELogitNorm 19, Mahalanobis→Mahalanobis 10, Mahalanobis→ReAct 8, Mahalanobis→ViM 6, Mahalanobis→Energy 4, Energy→ELogitNorm 4, ELogitNorm→ELogitNorm 3, Energy→ReAct 2, Energy→Energy 1, Mahalanobis→MSP 1, ReAct→MSP 1, ReAct→ViM 1; median tau-b 0.48; CI-significant swaps 2

## brain_cheng

(A) frozen M_std:

| arch | gap MSP | gap Energy | gap ELogitNorm | gap ViM | gap ReAct | gap Mahalanobis | gap kNN | Δfit Mahalanobis | Δfit kNN | Δfit ViM | Δfit ReAct | acc seen / unseen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_tiny | -0.018u | +0.021u | +0.066u | +0.015u | +0.346u | +0.000u | +0.001u | +0.000 | +0.000 | +0.014* | -0.000 | 0.986 / 1.000 |
| densenet121 | +0.003u | +0.061u | +0.089u | +0.078u | +0.034u | +0.000u | -0.003*u | +0.000 | +0.001* | +0.016* | +0.000 | 0.983 / 1.000 |
| effb3 | +0.001*u | +0.000u | +0.000u | +0.315*u | +0.000u | +0.000u | +0.000u | +0.000 | +0.000 | +0.023* | +0.000 | 0.979 / 1.000 |
| efficientnet_v2_s | -0.061*u | -0.037u | -0.031u | +0.040u | -0.040u | +0.000u | -0.004*u | -0.000 | -0.000 | +0.006 | -0.000 | 0.974 / 1.000 |
| mobilenet_v3_large | +0.117u | +0.149u | +0.165u | +0.010u | +0.118u | +0.000u | -0.003u | +0.000 | +0.000 | +0.011* | +0.000 | 0.987 / 1.000 |
| regnet_y_3_2gf | -0.031*u | -0.007*u | -0.003*u | +0.033u | -0.010*u | +0.000u | +0.000u | +0.000 | +0.000 | +0.014* | -0.000 | 0.984 / 1.000 |
| resnet18 | +0.159u | +0.162u | +0.214u | +0.033u | +0.128u | +0.000u | -0.003*u | +0.000 | +0.000 | +0.005 | +0.000 | 0.981 / 1.000 |
| resnet50 | +0.213u | +0.238u | +0.243u | +0.001u | +0.231u | +0.000u | +0.012u | +0.000 | +0.000 | +0.003* | -0.000 | 0.979 / 1.000 |

Ranking (A, 12 runs): Mahalanobis→Mahalanobis 11, Energy→Energy 1; median tau-b 0.75; CI-significant swaps 0

