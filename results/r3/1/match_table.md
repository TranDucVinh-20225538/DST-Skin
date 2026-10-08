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
