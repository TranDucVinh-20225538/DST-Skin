# Gate decision (L4)

**PASS**

- n loaded and scored FMs = 4 (dinov2_vitb14, uni, conch_v1_5, dinov2_vitl14)
- (1a) FMs with >= 2 of 3 counted feature Δ_fit > 0.02: 4 / 4 → True
- (1b) median of per-FM median feature Δ_fit = +0.071 (CNN anchor median +0.131), FMs with >= 2 CIs excluding 0: 4 / 4 → True
- (1) = True
- (2) median |Δ_fit ReAct| over 2 FMs = +0.000 (<= 0.02) → True

| FM | counted feature scores | # Δ_fit > 0.02 | inflation | median feature Δ_fit | # CI excl. 0 | ReAct Δ_fit |
|---|---|---|---|---|---|---|
| dinov2_vitb14 | Mahalanobis, kNN, ViM | 3 | True | +0.067 | 3 | +0.000 |
| uni | Mahalanobis, kNN, ViM | 2 | True | +0.049 | 3 | +0.000 |
| conch_v1_5 | Mahalanobis, kNN, ViM | 3 | True | +0.097 | 3 | +0.000 |
| dinov2_vitl14 | Mahalanobis, kNN, ViM | 3 | True | +0.074 | 3 | -0.000 |

## With Virchow2 (campaign addition, deviation 2; reported separately)

**PASS**

- n loaded and scored FMs = 5 (dinov2_vitb14, uni, conch_v1_5, dinov2_vitl14, virchow2)
- (1a) FMs with >= 2 of 3 counted feature Δ_fit > 0.02: 5 / 5 → True
- (1b) median of per-FM median feature Δ_fit = +0.067 (CNN anchor median +0.131), FMs with >= 2 CIs excluding 0: 5 / 5 → True
- (1) = True
- (2) median |Δ_fit ReAct| over 3 FMs = +0.000 (<= 0.02) → True

| FM | counted feature scores | # Δ_fit > 0.02 | inflation | median feature Δ_fit | # CI excl. 0 | ReAct Δ_fit |
|---|---|---|---|---|---|---|
| dinov2_vitb14 | Mahalanobis, kNN, ViM | 3 | True | +0.067 | 3 | +0.000 |
| uni | Mahalanobis, kNN, ViM | 2 | True | +0.049 | 3 | +0.000 |
| conch_v1_5 | Mahalanobis, kNN, ViM | 3 | True | +0.097 | 3 | +0.000 |
| dinov2_vitl14 | Mahalanobis, kNN, ViM | 3 | True | +0.074 | 3 | -0.000 |
| virchow2 | Mahalanobis, kNN, ViM | 3 | True | +0.053 | 3 | -0.000 |
