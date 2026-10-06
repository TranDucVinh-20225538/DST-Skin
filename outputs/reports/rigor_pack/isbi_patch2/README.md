# ISBI patch 2 readout

Precommit: `decisions/precommit_isbi_patch2_2026-10-06.md` (6477b25), committed before any patch-2 number.

## P1 ResNet50 0.27

No technical bug (`resnet50_diagnosis.md`): recompute matches, best = last epoch, no constant predictions on ID; the classifier collapses to 'normal' on hospital 2. Patch-1 numbers kept, labelled anomalous fold. Precision note: `calc_auroc` casts scores to float32; saturated MSP ties move the ResNet50 AUROC by <= 0.001 (same convention for all published numbers, not changed).

## P2 balanced folds

5 + 5 slides per hospital, min train-patch difference, ties by default_rng(20261006): train 172943 / 129493 patches (patch 1: 97,099 / 205,337); id_val 19,099 / 14,461. Exact balance is impossible under 5 + 5 because hospitals 3 and 4 each have one slide with > 50k patches.

## P3 grid and budget

Seeds run: 42, 43, 44. Projection after the smoke (ResNet50 s42 f0): 16.6 GPU-h for 18 runs -> seed 44 kept.
GPU-h estimate (precommit) 17-18 GPU-h (precommit: ~16 h training + ~1-2 h extraction); actual 16.5 GPU-h (train + extraction, 18 runs).

## P4 within-model gap (primary) and published delta (secondary)

| arch | n_seeds | gap_msp_mean | gap_msp_sd | gap_energy_mean | gap_energy_sd | delta_pub_msp_mean | delta_pub_energy_mean | over_bar |
|---|---|---|---|---|---|---|---|---|
| resnet50 | 3 | 0.160 | 0.025 | 0.141 | 0.021 | 0.223 | 0.227 | True |
| convnext_tiny | 3 | 0.193 | 0.013 | 0.178 | 0.020 | 0.249 | 0.249 | True |
| densenet121 | 3 | 0.185 | 0.029 | 0.178 | 0.033 | 0.396 | 0.398 | True |

Bar (gap > 0.02 for MSP or Energy on >= 2/3 archs, mean over seeds): **PASS: logit scores also benefit from slide sharing (3/3 archs > 0.02)**.

Unseen-slide accuracy per fold: fold 0 0.914-0.971; fold 1 0.644-0.757.
Confound note: unseen-slide accuracy < 0.8 in 9 fold(s) (resnet50 s42 f1 0.676, convnext_tiny s42 f1 0.707, densenet121 s42 f1 0.644, resnet50 s43 f1 0.657, convnext_tiny s43 f1 0.740, densenet121 s43 f1 0.757, resnet50 s44 f1 0.658, convnext_tiny s44 f1 0.703, densenet121 s44 f1 0.655); in those cells the gap mixes slide sharing with poor generalisation to new slides.
Descriptive (not a precommitted bar): gap range per fold, MSP / Energy: fold 0 0.107-0.178 / 0.080-0.157 (unseen acc 0.91-0.97); fold 1 0.136-0.268 / 0.128-0.272 (unseen acc 0.64-0.76).
Anomalous (ResNet50 < 0.5) cells: 6 of 6.

## P5 same-protocol H5(a) (all 7 scores, fit = fold train slides, ID = unseen slides)

| arch | cells | F3 | F2 |
|---|---|---|---|
| resnet50 | 6 | 6 | 5 |
| convnext_tiny | 6 | 6 | 5 |
| densenet121 | 6 | 6 | 5 |

Feature-space (F3) wins 18/18 cells; F2 15/18. Winners: Mahalanobis 15, ViM 3. Per cell: `same_protocol_h5a.csv`.

## Bug fixes

None in patch 2 (P1 found no bug).

