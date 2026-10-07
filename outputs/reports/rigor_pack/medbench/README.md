# Medbench group-leakage check: README

Precommit `decisions/precommit_group_leakage_medbench_2026-10-06.md` (commit 0fa52af, DEPOSIT 163151d),
deviations in `decisions/precommit_group_leakage_medbench_2026-10-06.deviations.md`. Bar = 0.02 AUROC.

## Bars per dataset

| dataset | archs passing gates (A) | A-bar present for (robust) | B-bar logit scores (robust) | ranking A: winner changed / sig swaps / median tau | ranking B | verdict (robust) |
|---|---|---|---|---|---|---|
| dermamnist | 8 | ViM, Mahalanobis (ViM, Mahalanobis) | none (none) | 1/0 of 8, 0.81 → stable (stable) | 0/0 of 4, 0.90 → stable (stable) | present (present) |
| isic2019 | 8 | ViM, Mahalanobis, kNN (ViM, Mahalanobis, kNN) | none (none) | 4/1 of 8, 0.81 → changed (stable) | 0/0 of 4, 0.81 → stable (stable) | present (present) |
| kermany | 8 | MSP, Energy, ELogitNorm, ViM, ReAct, Mahalanobis, kNN (MSP, Energy, ELogitNorm, ViM, ReAct, Mahalanobis, kNN) | none (none) | 8/8 of 8, 0.10 → changed (changed) | 0/0 of 4, 0.90 → stable (stable) | present (present) |
| breakhis | 8 | MSP, Energy, ELogitNorm, ViM, ReAct, Mahalanobis, kNN (ViM, Mahalanobis, kNN) | MSP, Energy, ELogitNorm (none) | 8/0 of 8, 0.48 → changed (changed) | — | present (present) |
| brain_cheng | 8 | none (none) | none (none) | 0/0 of 8, 0.75 → stable (stable) | — | absent, gap underpowered: Δ_fit only (absent) |

A-bar hit counts (archs with gap > 0.02 or Δ_fit > 0.02 / archs counted):

- dermamnist: MSP 0/8, Energy 0/8, ELogitNorm 2/8, ViM 7/8, ReAct 1/8, Mahalanobis 5/8, kNN 0/8
- isic2019: MSP 0/8, Energy 0/8, ELogitNorm 0/8, ViM 6/8, ReAct 0/8, Mahalanobis 8/8, kNN 4/8
- kermany: MSP 8/8, Energy 8/8, ELogitNorm 8/8, ViM 8/8, ReAct 8/8, Mahalanobis 8/8, kNN 8/8
- breakhis: MSP 8/8, Energy 8/8, ELogitNorm 8/8, ViM 8/8, ReAct 8/8, Mahalanobis 8/8, kNN 8/8
- brain_cheng: MSP 0/0, Energy 0/0, ELogitNorm 0/0, ViM 1/8, ReAct 0/8, Mahalanobis 0/8, kNN 0/8

B-bar hit counts (core archs with within-model gap > 0.02 / counted):

- dermamnist: MSP 2/4, Energy 2/4, ELogitNorm 2/4, ViM 2/4, ReAct 1/4, Mahalanobis 1/4, kNN 1/4
- isic2019: MSP 2/4, Energy 2/4, ELogitNorm 0/4, ViM 1/4, ReAct 0/4, Mahalanobis 2/4, kNN 1/4
- kermany: MSP 0/4, Energy 0/4, ELogitNorm 0/4, ViM 2/4, ReAct 0/4, Mahalanobis 4/4, kNN 3/4
- breakhis: MSP 4/4, Energy 4/4, ELogitNorm 4/4, ViM 4/4, ReAct 4/4, Mahalanobis 4/4, kNN 4/4

Δ_fit alone (scorer fit on same vs other groups inside the seen set; no seen/unseen population difference),
archs with Δ_fit > 0.02 / counted, median Δ_fit:

- dermamnist: Mahalanobis 5/8 (+0.023), kNN 0/8 (+0.007), ViM 7/8 (+0.035), ReAct 0/8 (+0.000)
- isic2019: Mahalanobis 8/8 (+0.040), kNN 4/8 (+0.020), ViM 6/8 (+0.037), ReAct 0/8 (-0.000)
- kermany: Mahalanobis 8/8 (+0.025), kNN 6/8 (+0.026), ViM 6/8 (+0.030), ReAct 0/8 (+0.000)
- breakhis: Mahalanobis 8/8 (+0.193), kNN 8/8 (+0.154), ViM 8/8 (+0.111), ReAct 0/8 (+0.000)
- brain_cheng: Mahalanobis 0/8 (+0.000), kNN 0/8 (+0.000), ViM 1/8 (+0.012), ReAct 0/8 (-0.000)

## Generalization verdict (locked rule, with Camelyon / multibench rows)

Core medical datasets tested: 4 (dermamnist, isic2019, kermany, breakhis); leakage present on 4 (robust bar: 4) → **group leakage in standard medical splits inflates post-hoc OOD AUROC across benchmarks** (robust: group leakage in standard medical splits inflates post-hoc OOD AUROC across benchmarks).
Earlier rows: Camelyon17 present; multibench iWildCam present, RxRx1 present (A only), MIDOG absent by construction.

## Secondary directional prediction

- dermamnist: median A-gap feature family -0.009 vs logit family -0.014; ReAct -0.007
- isic2019: median A-gap feature family -0.077 vs logit family -0.085; ReAct -0.082
- kermany: median A-gap feature family +0.368 vs logit family +0.165; ReAct +0.154
- breakhis: median A-gap feature family +0.154 vs logit family +0.136; ReAct +0.117
Feature > logit on 4/4 (prediction needs >= 3/4): holds; ReAct |gap| <= 0.02 on 1/4.

## Model-quality gates

- dermamnist: 36/36 runs pass
- isic2019: 28/28 runs pass
- kermany: 36/36 runs pass
- breakhis: 83/100 runs pass: failing convnext_tiny s42 gd_r0 (G1 True, G2 True = 0.565, G3 near-chance MSP,Energy,ELogitNorm,ReAct,kNN); convnext_tiny s42 gd_r3 (G1 True, G2 False = 0.330, G3 near-chance ELogitNorm,ViM,Mahalanobis,kNN); convnext_tiny s43 gd_r3 (G1 True, G2 False = 0.377, G3 near-chance ELogitNorm,Mahalanobis,kNN); densenet121 s42 gd_r3 (G1 True, G2 False = 0.301, G3 near-chance Energy,ELogitNorm,ViM,ReAct,Mahalanobis,kNN); densenet121 s43 gd_r3 (G1 True, G2 False = 0.339, G3 near-chance Energy,ELogitNorm,ViM,ReAct,Mahalanobis,kNN); resnet18 s42 gd_r0 (G1 True, G2 True = 0.551, G3 near-chance MSP,Energy,ELogitNorm,ViM,ReAct,kNN); resnet18 s42 gd_r1 (G1 True, G2 False = 0.489, G3 near-chance MSP,Energy,ELogitNorm,ViM,ReAct); resnet18 s42 gd_r3 (G1 True, G2 False = 0.314, G3 near-chance Energy,ELogitNorm,ViM,ReAct,Mahalanobis); resnet18 s42 gd_r4 (G1 True, G2 True = 0.683, G3 near-chance MSP,Energy,ELogitNorm,ViM,ReAct,Mahalanobis,kNN); resnet18 s43 gd_r0 (G1 True, G2 True = 0.525, G3 near-chance MSP,Energy,ViM,ReAct,kNN); resnet18 s43 gd_r1 (G1 True, G2 True = 0.523, G3 near-chance MSP,Energy,ELogitNorm,ViM,ReAct); resnet18 s43 gd_r3 (G1 True, G2 False = 0.341, G3 near-chance Energy,ViM,ReAct,Mahalanobis,kNN); resnet18 s43 gd_r4 (G1 True, G2 True = 0.684, G3 near-chance MSP,Energy,ELogitNorm,ViM,ReAct,Mahalanobis,kNN); resnet50 s42 gd_r3 (G1 True, G2 False = 0.321, G3 near-chance ELogitNorm,ViM,kNN); resnet50 s43 gd_r0 (G1 True, G2 True = 0.556, G3 near-chance MSP,Energy,ViM,ReAct,Mahalanobis,kNN); resnet50 s43 gd_r1 (G1 True, G2 True = 0.535, G3 near-chance MSP,Energy,ELogitNorm,ViM,ReAct); resnet50 s43 gd_r3 (G1 True, G2 False = 0.347, G3 near-chance Energy,ViM,ReAct,Mahalanobis,kNN)
- brain_cheng: 12/12 runs pass

Confound notes (unseen-group accuracy < 0.8):

- (A) breakhis/convnext_tiny 0.684, breakhis/densenet121 0.677, breakhis/effb3 0.678, breakhis/efficientnet_v2_s 0.660, breakhis/mobilenet_v3_large 0.653, breakhis/regnet_y_3_2gf 0.673, breakhis/resnet18 0.651, breakhis/resnet50 0.676
- (B) isic2019/convnext_tiny 0.748, isic2019/densenet121 0.737, isic2019/resnet18 0.728, isic2019/resnet50 0.733, breakhis/convnext_tiny 0.684, breakhis/densenet121 0.677, breakhis/resnet18 0.651, breakhis/resnet50 0.676

## Budget

Estimate (precommit L6): ≈ 70 GPU-h core + ≈ 3 brain. Actual (sum of medbench GPU job elapsed, smokes included): 40.1 GPU-h.
No cut was needed.

## Sensitivity readouts (reported only, not in any bar)

Median over archs × seeds (CI-excluding-0 runs / runs in brackets):

| dataset | readout | n unseen (median) | gap MSP | gap Energy | gap ELogitNorm | gap ViM | gap ReAct | gap Mahalanobis | gap kNN | Δfit Maha / kNN / ViM |
|---|---|---|---|---|---|---|---|---|---|---|
| isic2019 | with_lesion_id | 1533 | -0.111 (0/12) | -0.104 (0/12) | -0.095 (0/12) | -0.080 (0/12) | -0.101 (0/12) | -0.099 (0/12) | -0.117 (0/12) | +0.039 / +0.020 / +0.037 |
| kermany | unseen_v2_only | 86 | +0.022 (0/12) | +0.004 (0/12) | -0.001 (0/12) | +0.022 (0/12) | +0.006 (0/12) | +0.035 (0/12) | +0.042 (0/12) | +nan / +nan / +nan |
| kermany | unseen_v3_only | 750 | +0.171 (12/12) | +0.188 (12/12) | +0.184 (12/12) | +0.258 (12/12) | +0.165 (12/12) | +0.479 (12/12) | +0.415 (12/12) | +nan / +nan / +nan |

## Descriptive arms (not in bars)

- Kermany M_std(v3) on the official v3 test (patient-disjoint), AUROC vs v3 DRUSEN, mean of 8 runs: MSP 0.550, Energy 0.540, ELogitNorm 0.581, ViM 0.327, ReAct 0.482, Mahalanobis 0.217, kNN 0.331; acc 0.968.
- DermaMNIST-C arm (8 runs), AUROC vs PAD-UFES-20, C test (lesion-disjoint) / E test (ISIC 2018 test): MSP 0.858 / 0.760, Energy 0.912 / 0.857, ELogitNorm 0.893 / 0.853, ViM 0.898 / 0.838, ReAct 0.885 / 0.814, Mahalanobis 0.961 / 0.928, kNN 0.930 / 0.875; acc 0.899 / 0.800.
- BreakHis secondary AUROC(M_std, leaky ID_seen) − AUROC(M_gd, P_out), all 40 M_gd runs (23 pass the gates; M_gd balanced acc on P_out 0.540): MSP +0.156, Energy +0.161, ELogitNorm +0.132, ViM +0.102, ReAct +0.132, Mahalanobis +0.174, kNN +0.162.
- Brain (optional, M_std only): ID_unseen 2 images / 2 patients → underpowered, counts toward no bar; (B) cvind arm not run.

## Population confounds of the A-gap (descriptive)

- ISIC 2019 source mix: test_seen BCN 75%, HAM 25%, MSK 0%, no_lesion_id 0%; test_unseen BCN 9%, HAM 62%, MSK 8%, no_lesion_id 21%; OOD BCN 48%, HAM 52%, MSK 0%, no_lesion_id 0%. ID_seen is mostly BCN, ID_unseen mostly HAM / no-id, so the A-gap mixes leakage with a source shift; dropping no-id images does not remove it (sensitivity table).
- Kermany: ID_unseen = 86 v2-test + 750 v3-test images. The v3 supplement drives the A-gap (sensitivity table); M_std(v3) also scores its own official v3 test below the v3 DRUSEN OOD set (descriptive arms), i.e. the v3 test images are shifted from training images independently of group sharing.
- BreakHis: ID_unseen = P_out patients, unseen-patient accuracy ≈ 0.65-0.68 vs ≈ 0.9 on ID_seen (confound flag); the A-gap includes the patient-shift drop in accuracy.

## Aggregation choices (technical; not in the precommit text)

- Gates per run; failing runs dropped; an arch counts if >= 1 run passes; arch value = mean over passing runs.
- G2 set: ID_seen of the arm (`test_seen` / `seen`); DermaMNIST-C arm: `c_test`; v3std and BreakHis M_gd: their only ID set.
- G3 on class-matched seen / unseen AUROCs (arms without them: on the unmatched AUROCs).
- Floor: A-gap / B-gap cells use ID_unseen (BreakHis: summed over repeats, the pooled cell of deviation 4);
  Δ_fit cells use the smaller seen-set fold.
- Robust bar: point > 0.02 and every passing run's CI excludes 0.
- Ranking: leaky = class-matched AUROC on ID_seen, disjoint = on ID_unseen; near-chance scores dropped from the
  ranking of that run; a swap is CI-significant if the leaky winner beats the disjoint winner under the leaky
  protocol and the reverse holds under the disjoint protocol (both pair CIs exclude 0); per arch = majority of runs.
- BreakHis (B) = within-model gap of M_std(r), core archs (L3); M_gd(r) reported in model_quality.csv only.

