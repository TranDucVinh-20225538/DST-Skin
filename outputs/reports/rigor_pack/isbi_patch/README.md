# ISBI patch readout

Precommit: `decisions/precommit_isbi_patch_2026-10-05.md` (committed before any number below).

## A. ViM / ReAct slide-disjoint 2-fold (bar: any arch loss > 0.02)

| score | max_loss | arch_max_loss | n_archs_loss_gt_0.02 | n_archs | bar |
|---|---|---|---|---|---|
| vim | 0.092 | densenet121 | 8 | 8 | appendix + H5(a) |
| react | 0.005 | effb3 | 0 | 8 | one sentence (<= 0.02) |

Per arch: `leakage/leakfree_vim_react.csv`.

## B. Slide-disjoint logit retrain (MSP/Energy)

Bar reading: **not a fair same-protocol comparison without noting training-slide overlap (3/3 archs > 0.02)**

| arch | folds_done | msp_published | msp_slide_disjoint_retrain | msp_same_slide_retrain | delta_msp | energy_published | energy_slide_disjoint_retrain | energy_same_slide_retrain | delta_energy | gpu_hours |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 2 | 0.515 | 0.272 | 0.519 | 0.243 | 0.512 | 0.267 | 0.509 | 0.245 | 0.965 |
| convnext_tiny | 2 | 0.846 | 0.562 | 0.762 | 0.284 | 0.877 | 0.609 | 0.814 | 0.268 | 2.883 |
| densenet121 | 2 | 0.883 | 0.609 | 0.791 | 0.274 | 0.877 | 0.605 | 0.783 | 0.273 | 1.388 |

GPU-hours: precommit estimate 9-12 GPU-h, actual 5.2 GPU-h (sum of job wall times of the finished folds).

## C. H5(a) re-rank, seed 42 (feature family primary {Maha, kNN, ViM}; secondary {Maha, kNN})

| reading | n | F3 | F2 |
|---|---|---|---|
| i_original | 8 | 8 | 8 |
| ii_leakfree_maha_knn | 8 | 4 | 4 |
| iii_leakfree_maha_knn_vim_react | 8 | 5 | 5 |
| iv_plus_logit_retrain | 3 | 2 | 2 |

Per arch: `h5a_rerank.csv`. Locked wording: "feature-space wins on K/8 backbones under slide-disjoint scoring" with K = 5 (reading iii, primary family); never "feature always best".

## D. Disjoint-partition sample size

| tree | axis | min_reliable_disjoint_partition | min_reliable_overlapping_subset | sizes_tested_disjoint |
|---|---|---|---|---|
| stable | seed | 1 | 1 | 1..4 |
| stable | arch | 6 | 2 | 2..6 |
| stable | id_patients | 2 | 2 | 2..24 |
| stable | ood_patients | 2 | 2 | 2..7 |
| stable_maha_vim8 | seed | 1 | 1 | 1..4 |
| stable_maha_vim8 | arch | không đạt trong dữ liệu hiện có | 4 | 2..6 |
| stable_maha_vim8 | id_patients | 2 | 2 | 2..24 |
| stable_maha_vim8 | ood_patients | 7 | 2 | 2..7 |

Sizes where overlapping subsets passed but disjoint partitions failed (manuscript must prefer the disjoint numbers there): stable grid 1x3; stable grid 1x4; stable grid 1x5; stable grid 2x3; stable grid 2x4; stable grid 2x5; stable grid 3x3; stable grid 3x4; stable grid 3x5; stable grid 4x3; stable grid 4x4; stable grid 4x5; stable_maha_vim8 grid 1x5; stable_maha_vim8 grid 1x6; stable_maha_vim8 grid 2x5; stable_maha_vim8 grid 2x6; stable_maha_vim8 grid 3x4; stable_maha_vim8 grid 3x5; stable_maha_vim8 grid 3x6; stable_maha_vim8 grid 4x4; stable_maha_vim8 grid 4x5; stable_maha_vim8 grid 4x6; stable arch 3; stable arch 4; stable arch 5; stable_maha_vim8 arch 4; stable_maha_vim8 arch 5; stable_maha_vim8 arch 6; stable_maha_vim8 ood_patients 3; stable_maha_vim8 ood_patients 6

## E. H15 checklist

`checklist_h15.md`: robust = ConvNeXt, MobileNet.

## F. Scope

`scope_table.md` (Camelyon17 seed-42 leak-free deltas + MIDOG H12 one-liner).

Suggested wording and caveats (fold sizes, fold-0 accuracy on unseen slides): `claim_edits.md`.
