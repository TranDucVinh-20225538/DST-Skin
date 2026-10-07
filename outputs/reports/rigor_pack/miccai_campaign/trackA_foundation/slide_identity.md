# Slide-identity mechanism check (Camelyon, seed 42; post-hoc, descriptive)

Same 30 training slides and folds for every backbone. Probe: balanced accuracy of a linear slide-ID probe (300 / 100 patches per slide, chance 1/30). ICC: between-slide / total variance of L2-normalised features.

```
             model kind  feat_dim  slide_probe_bacc   icc  dfit_feature_median  dfit_Mahalanobis  dfit_kNN  dfit_ViM  dfit_ReAct
               uni   FM      1024             0.931 0.252                0.049             0.008     0.093     0.049       0.000
          virchow2   FM      2560             0.925 0.239                0.053             0.033     0.112     0.053      -0.000
     convnext_tiny  CNN       768             0.816 0.413                0.059             0.059     0.054     0.074       0.002
     dinov2_vitb14   FM       768             0.720 0.123                0.067             0.067     0.112     0.028       0.000
     dinov2_vitl14   FM      1024             0.767 0.110                0.074             0.074     0.107     0.029      -0.000
 efficientnet_v2_s  CNN      1280             0.748 0.436                0.086             0.083     0.086     0.086       0.002
mobilenet_v3_large  CNN       960             0.744 0.325                0.091             0.099     0.091     0.055      -0.001
       densenet121  CNN      1024             0.765 0.451                0.093             0.113     0.093     0.078       0.000
        conch_v1_5   FM       768             0.787 0.297                0.097             0.097     0.150     0.042       0.000
    regnet_y_3_2gf  CNN      1512             0.787 0.277                0.122             0.149     0.122     0.080      -0.001
          resnet18  CNN       512             0.735 0.523                0.136             0.168     0.136     0.108       0.002
             effb3  CNN      1536             0.749 0.482                0.136             0.140     0.136     0.018       0.005
          resnet50  CNN      2048             0.739 0.503                0.203             0.216     0.203     0.074      -0.000
```

- Δ_fit vs slide_probe_bacc, all (n = 13), spearman r = -0.527, permutation p = 0.0716
- Δ_fit vs slide_probe_bacc, all (n = 13), pearson r = -0.550, permutation p = 0.0583
- Δ_fit vs slide_probe_bacc, CNN (n = 8), spearman r = -0.500, permutation p = 0.2151
- Δ_fit vs slide_probe_bacc, CNN (n = 8), pearson r = -0.565, permutation p = 0.1537
- Δ_fit vs slide_probe_bacc, FM (n = 5), spearman r = -0.600, permutation p = 0.3407
- Δ_fit vs slide_probe_bacc, FM (n = 5), pearson r = -0.659, permutation p = 0.2572
- Δ_fit vs icc, all (n = 13), spearman r = +0.703, permutation p = 0.0089
- Δ_fit vs icc, all (n = 13), pearson r = +0.622, permutation p = 0.0200
- Δ_fit vs icc, CNN (n = 8), spearman r = +0.619, permutation p = 0.1173
- Δ_fit vs icc, CNN (n = 8), pearson r = +0.420, permutation p = 0.3057
- Δ_fit vs icc, FM (n = 5), spearman r = +0.000, permutation p = 1.0000
- Δ_fit vs icc, FM (n = 5), pearson r = +0.120, permutation p = 0.9320
