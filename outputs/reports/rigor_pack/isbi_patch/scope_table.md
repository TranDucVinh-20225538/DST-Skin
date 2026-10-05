# Scope table (Limitations)

Camelyon17, seed 42. d_* = slide-disjoint minus same-slide AUROC (2-fold, H11c folds); d_*_retrain = slide-disjoint retrain minus published (Phase B; '-' = not in the grid).

| arch | d_Maha | d_kNN | d_ViM | d_ReAct | d_MSP_retrain | d_Energy_retrain |
|---|---|---|---|---|---|---|
| R18 | -0.168 | -0.136 | -0.080 | -0.002 | - | - |
| R50 | -0.209 | -0.203 | -0.073 | 0.000 | -0.243 | -0.245 |
| DenseNet | -0.113 | -0.093 | -0.092 | -0.000 | -0.274 | -0.273 |
| ConvNeXt | -0.059 | -0.054 | -0.074 | -0.002 | -0.284 | -0.268 |
| MobileNet | -0.099 | -0.091 | -0.055 | 0.001 | - | - |
| RegNet | -0.149 | -0.122 | -0.080 | 0.001 | - | - |
| EffB3 | -0.141 | -0.136 | -0.036 | -0.005 | - | - |
| EffV2-S | -0.083 | -0.086 | -0.086 | -0.002 | - | - |

MIDOG (H12, already computed, CSV only): midog: H1 mixed, H4 arch effect detected

No other domain or setup is covered by these checks.
