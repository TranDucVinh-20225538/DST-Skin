# Campaign summary (precommit `decisions/precommit_miccai_full_campaign_2026-10-07.md`, f438005)

All numbers from the files listed in `inventory.md`. Δ_fit = AUROC(scorer fit on same groups) − AUROC(scorer fit
on disjoint groups), 2-fold, seed 42 unless stated. Feature family = Mahalanobis / kNN / ViM.

## Track A — Δ_fit (feature family, median over backbones)

| dataset | all backbones | CNN | FM | n backbones (FM) | ranking rows changed / n | median τ-b |
|---|---|---|---|---|---|---|
| Camelyon17 | +0.091 | +0.108 | +0.067 | 13 (5) | 5 / 13 | +0.714 |
| BreakHis | +0.151 | +0.155 | +0.078 | 13 (5) | 39 / 65 | +0.524 |
| ISIC 2019 | +0.026 | +0.034 | +0.020 | 13 (5) | 12 / 31 | +0.810 |
| DermaMNIST | +0.017 | +0.021 | +0.010 | 12 (4) | 1 / 30 | +0.857 |
| Kermany | +0.030 | +0.030 | NA (no FM) | 8 (0) | 8 / 16 | +0.533 |

Camelyon 3-seed check (anchors + FMs, seeds 42-44, n = 21): +0.070. ReAct median |Δ_fit| on Camelyon FM cells: 0.000.
CI-significant winner swaps (seed 42, `swap_ci_significant`): BreakHis 2 / 65 rows (39 changed), ISIC 2019 3 / 13
family-A + 0 / 18 family-B, DermaMNIST 0 / 30, Kermany 8 / 8 family-A (v2 + v3 pooled new-patient set, confounded)
+ 0 / 8 family-B.

## Foundation gate (L4): PASS

| FM | median feature Δ_fit | # Δ > 0.02 | # CI excl. 0 (cluster bootstrap) | ReAct |
|---|---|---|---|---|
| DINOv2-B | +0.067 | 3 | 3 | 0.000 |
| UNI | +0.049 | 2 | 3 | 0.000 |
| CONCH v1.5 | +0.097 | 3 | 3 | 0.000 |
| DINOv2-L | +0.074 | 3 | 3 | 0.000 |
| Virchow2 (extra) | +0.053 | 3 | 3 | 0.000 |

Median of FM medians +0.071 (4 precommitted FMs), +0.067 with Virchow2; CNN anchor median +0.131.

## Track C — τ at 95% TPR on leaky ID, realized TPR on new-group ID (median over backbones; descriptive)

| dataset | Mahalanobis | kNN | ViM | MSP | Energy | ReAct |
|---|---|---|---|---|---|---|
| Camelyon17 | 0.886 | 0.791 | 0.895 | 0.950 | 0.950 | 0.948 |
| BreakHis | 0.868 | 0.868 | 0.894 | 0.886 | 0.846 | 0.850 |
| ISIC 2019 | 0.971 | 0.971 | 0.971 | 0.978 | 0.979 | 0.981 |
| DermaMNIST | 0.963 | 0.967 | 0.963 | 0.966 | 0.967 | 0.964 |
| Kermany (pooled v2 + v3) | 0.444 | 0.538 | 0.798 | 0.714 | 0.715 | 0.714 |

Kermany split (post-hoc, `trackC_clinical_tau/kermany_confound.md`): v2 new patients (n = 86) Maha 0.890 / kNN 0.878;
v3 supplement (n = 750) Maha 0.392 / kNN 0.495; accuracy 0.97-1.00 on both; class matching changes nothing.
Risk-coverage: skipped (`trackC_clinical_tau/risk_coverage_skip.md`).

## Track D — scorer-fit vs backbone leakage (Camelyon, isbi_patch2 retrain arm)

Locked table: `trackD_backbone_leak/side_by_side.csv`. Post-hoc two-channel table (`two_channel.md`), fold 0
(unseen-slide accuracy 0.91-0.97), within-model seen − unseen AUROC, mean over seeds 42-44:

| arch | MSP | Energy | ReAct | Maha | kNN | ViM | Track A Δ_fit feature, seed 42 | Track A, median over seeds (n) |
|---|---|---|---|---|---|---|---|---|
| ResNet50 | +0.140 | +0.103 | +0.279 | +0.059 | +0.158 | +0.021 | +0.203 | +0.078 (3) |
| ConvNeXt-T | +0.137 | +0.114 | +0.186 | +0.102 | +0.131 | +0.090 | +0.059 | +0.071 (3) |
| DenseNet121 | +0.153 | +0.146 | +0.196 | +0.075 | +0.142 | +0.069 | +0.093 | +0.093 (1) |

Fold 1 (unseen-slide accuracy 0.66-0.72, confound flag) is larger for every score.

## Track E — F2 (cross-fit) vs F1 (disjoint 2-fold)

| dataset | median |F2 − F1| | median τ-b | n backbones | works |
|---|---|---|---|---|
| Camelyon17 | 0.016 | 0.905 | 13 | yes |
| BreakHis | 0.002 | 1.000 | 13 | yes |
| ISIC 2019 | 0.000 | 1.000 | 13 | yes |
| DermaMNIST | 0.000 | 1.000 | 13 | yes |
| Kermany | 0.000 | 1.000 | 8 | yes |

F2 works on 5 / 5 datasets tested (bar: ≥ 2/3).

## Mechanism check (post-hoc, Camelyon, 13 backbones; `trackA_foundation/slide_identity.md`)

Δ_fit vs between-slide variance share (ICC): Spearman ρ = +0.70 (permutation p = 0.009); within CNN +0.62 (p = 0.12),
within FM 0.00. Δ_fit vs linear slide-probe balanced accuracy: ρ = −0.53 (p = 0.07). 12 tests, not corrected.

## Tracks not run

B: STOP (`trackB_new_medical/STOP_reason.md`). F: skipped, MIDOG case IDs not recoverable
(`trackF_openmibood/skip_reason.md`). I: not started (`trackI_stain/followup_stain_ablation_note.md`).

## Venue

Precommit L7 readout: MICCAI (`decision_miccai_vs_midl.md`, internal). Chosen venue: MIDL 2027 (`venue_note.md`).
