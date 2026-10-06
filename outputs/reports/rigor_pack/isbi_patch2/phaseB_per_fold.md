# Phase B v2 per fold (balanced folds, seeds 42, 43, 44)

Primary = within-model gap (seen-slide minus unseen-slide id_val AUROC, same model, OOD = hospital 2). Secondary = published seed-42 AUROC minus unseen-slide retrain AUROC. `anomalous` = ResNet50 cell below 0.5 (P1: no bug, classifier collapse on hospital 2).

| arch | seed | fold | acc_id_seen | acc_id_unseen | acc_ood | auroc_msp_unseen | auroc_msp_seen | gap_msp | auroc_energy_unseen | auroc_energy_seen | gap_energy | delta_published_msp | anomalous |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 42 | 0 | 0.994 | 0.914 | 0.500 | 0.229 | 0.336 | 0.107 | 0.218 | 0.300 | 0.082 | 0.286 | True |
| resnet50 | 42 | 1 | 0.998 | 0.676 | 0.563 | 0.433 | 0.643 | 0.210 | 0.425 | 0.635 | 0.211 | 0.082 | True |
| convnext_tiny | 42 | 0 | 0.996 | 0.957 | 0.767 | 0.696 | 0.831 | 0.135 | 0.725 | 0.854 | 0.129 | 0.150 | False |
| convnext_tiny | 42 | 1 | 0.997 | 0.707 | 0.772 | 0.454 | 0.722 | 0.268 | 0.523 | 0.795 | 0.272 | 0.392 | False |
| densenet121 | 42 | 0 | 0.996 | 0.914 | 0.517 | 0.552 | 0.702 | 0.150 | 0.542 | 0.683 | 0.141 | 0.331 | False |
| densenet121 | 42 | 1 | 0.997 | 0.644 | 0.499 | 0.306 | 0.460 | 0.154 | 0.290 | 0.429 | 0.139 | 0.576 | False |
| resnet50 | 43 | 0 | 0.994 | 0.915 | 0.505 | 0.174 | 0.352 | 0.178 | 0.164 | 0.282 | 0.118 | 0.341 | True |
| resnet50 | 43 | 1 | 0.997 | 0.657 | 0.513 | 0.365 | 0.560 | 0.195 | 0.368 | 0.568 | 0.200 | 0.150 | True |
| convnext_tiny | 43 | 0 | 0.996 | 0.967 | 0.851 | 0.674 | 0.822 | 0.148 | 0.749 | 0.829 | 0.080 | 0.171 | False |
| convnext_tiny | 43 | 1 | 0.998 | 0.740 | 0.785 | 0.516 | 0.767 | 0.251 | 0.534 | 0.781 | 0.247 | 0.330 | False |
| densenet121 | 43 | 0 | 0.996 | 0.921 | 0.516 | 0.570 | 0.722 | 0.153 | 0.557 | 0.698 | 0.141 | 0.313 | False |
| densenet121 | 43 | 1 | 0.997 | 0.757 | 0.573 | 0.545 | 0.792 | 0.248 | 0.535 | 0.788 | 0.252 | 0.338 | False |
| resnet50 | 44 | 0 | 0.995 | 0.916 | 0.485 | 0.257 | 0.393 | 0.136 | 0.250 | 0.358 | 0.108 | 0.258 | True |
| resnet50 | 44 | 1 | 0.997 | 0.658 | 0.504 | 0.294 | 0.429 | 0.136 | 0.286 | 0.414 | 0.128 | 0.221 | True |
| convnext_tiny | 44 | 0 | 0.996 | 0.971 | 0.794 | 0.711 | 0.840 | 0.129 | 0.714 | 0.849 | 0.134 | 0.134 | False |
| convnext_tiny | 44 | 1 | 0.998 | 0.703 | 0.774 | 0.527 | 0.754 | 0.227 | 0.523 | 0.728 | 0.205 | 0.319 | False |
| densenet121 | 44 | 0 | 0.995 | 0.915 | 0.503 | 0.598 | 0.755 | 0.157 | 0.602 | 0.759 | 0.157 | 0.285 | False |
| densenet121 | 44 | 1 | 0.997 | 0.655 | 0.508 | 0.348 | 0.598 | 0.250 | 0.347 | 0.586 | 0.239 | 0.534 | False |

## Per arch: fold mean, then mean ± SD over seeds

| arch | n_seeds | gap_msp_mean | gap_msp_sd | gap_energy_mean | gap_energy_sd | delta_pub_msp_mean | delta_pub_msp_sd | delta_pub_energy_mean | delta_pub_energy_sd | over_bar |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 3 | 0.160 | 0.025 | 0.141 | 0.021 | 0.223 | 0.034 | 0.227 | 0.031 | True |
| convnext_tiny | 3 | 0.193 | 0.013 | 0.178 | 0.020 | 0.249 | 0.022 | 0.249 | 0.012 | True |
| densenet121 | 3 | 0.185 | 0.029 | 0.178 | 0.033 | 0.396 | 0.065 | 0.398 | 0.065 | True |
