# Predictor (P)

Unit = (dataset, arch, fit score), n = 80 cells over 3 datasets; target Δ (seed 42).
Bar: |Spearman ρ| >= 0.6 AND leave-one-dataset-out MAE < 0.03.

| candidate | ρ (all cells) | LODO MAE | MAE per held-out | ρ within dataset | verdict |
|---|---|---|---|---|---|
| M2_balanced_acc | +0.36 | 0.073 | camelyon 0.055, iwildcam 0.122, rxrx1 0.043 | camelyon -0.25, iwildcam +0.08, rxrx1 +0.00 | not predictive |
| M3_same_group_nn_frac | +0.24 | 0.081 | camelyon 0.085, iwildcam 0.121, rxrx1 0.035 | camelyon -0.13, iwildcam -0.03, rxrx1 -0.18 | not predictive |
| centroid_distance | +0.27 | 0.161 | camelyon 0.049, iwildcam 0.375, rxrx1 0.059 | camelyon -0.22, iwildcam -0.16, rxrx1 -0.18 | not predictive |
