# Track D two-channel table (post-hoc, descriptive)

Within-model seen - unseen AUROC on the isbi_patch2 slide-disjoint retrained models (mean over seeds 42-44 and the listed folds; scorer fitted on the fold's train slides, float64). Logit gap = backbone channel; feature gap = backbone + scorer-fit channels. Last numeric column: Track A scorer-fit-only Δ_fit (published backbone, median over available seeds). Fold 1: unseen-slide accuracy < 0.8 in every cell (confound).

| arch | folds | acc_unseen | gap_MSP | gap_Energy | gap_ReAct | gap_Mahalanobis | gap_kNN | gap_ViM | logit_gap_median | feature_gap_median | scorer_fit_dfit_feature_median_trackA | any_below_chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | both folds | 0.789 | +0.160 | +0.141 | +0.251 | +0.091 | +0.199 | +0.045 | +0.141 | +0.091 | +0.078 | MSP,Energy,ELogitNorm |
| convnext_tiny | both folds | 0.841 | +0.193 | +0.178 | +0.158 | +0.166 | +0.197 | +0.118 | +0.193 | +0.166 | +0.059 |  |
| densenet121 | both folds | 0.801 | +0.185 | +0.178 | +0.216 | +0.108 | +0.190 | +0.082 | +0.183 | +0.108 | +0.093 | MSP,Energy,ELogitNorm |
| resnet50 | fold 0 (acc >= 0.8) | 0.915 | +0.140 | +0.103 | +0.279 | +0.059 | +0.158 | +0.021 | +0.103 | +0.059 | +0.078 | MSP,Energy,ELogitNorm |
| convnext_tiny | fold 0 (acc >= 0.8) | 0.965 | +0.137 | +0.114 | +0.186 | +0.102 | +0.131 | +0.090 | +0.137 | +0.102 | +0.059 |  |
| densenet121 | fold 0 (acc >= 0.8) | 0.917 | +0.153 | +0.146 | +0.196 | +0.075 | +0.142 | +0.069 | +0.151 | +0.075 | +0.093 |  |
| resnet50 | fold 1 (acc < 0.8) | 0.664 | +0.180 | +0.180 | +0.224 | +0.124 | +0.239 | +0.068 | +0.180 | +0.124 | +0.078 | MSP,Energy,ELogitNorm |
| convnext_tiny | fold 1 (acc < 0.8) | 0.717 | +0.249 | +0.241 | +0.131 | +0.230 | +0.263 | +0.146 | +0.249 | +0.230 | +0.059 |  |
| densenet121 | fold 1 (acc < 0.8) | 0.685 | +0.217 | +0.210 | +0.237 | +0.141 | +0.238 | +0.096 | +0.214 | +0.141 | +0.093 | MSP,Energy,ELogitNorm |
