# Item A — closed form vs ICC on real cached features (REPORT)

**Verdict: NO-GO** (read off the A decision table; A-i fail, A-ii fail, A-iii fail, A-iv fail).

**Interpretation limits (binding wording).** (1) 13 backbones is a small number of independent units; the LOBO result is a screening result and its confidence intervals may be unstable. (2) The cluster bootstrap with B = 10,000 over backbones resamples the same 13 backbones; it does not create independent backbones and cannot make the interval narrower than 13 units allow. (3) The within-FM analysis has n = 5 units (within-CNN n = 8): it is descriptive only, has no power, and is not used to claim generalisation within either family. (4) The NO-GO says that, on these 13 backbones, the formula did not predict held-out backbones better than the stated baselines; the saturation ratio r_sat and the A-ii / A-iii rules are as pre-registered.

Consequence written in the order for NO-GO: the Gaussian closed form is not supported as the mechanism of real-feature leakage in this set; Idea 2 is downgraded to toy theory; G2 is skipped (G-ii `not run`); C-core and D run at their listed scope; C5's real-cell validity map is skipped (C5 on synthetic remains).

- Commit at report time: `9e81c880567c657fa27516deabaaa19a9a268cf5`; precommit commit `e326d392fd42ced26fbebe6c0a38ed983267d187`; work list: `results/t1/PRECOMMIT_T1_addendum_A.json` (8,116 units; all executed).
- GPU-hours (sacct, elapsed × GPUs): A0 0.10, A1 0.74, A2 0.57, A3 0.68; total **2.09** vs estimate 4–8 (re-derived 3–6), cap 12.
- Wall clock: first job start 2026-10-11T02:58:21, last job end 2026-10-11T04:08:25.

## Status words

| sub-criterion | status | numbers |
|---|---|---|
| A-i (A1 vs B1 and B1w, Camelyon LOBO, n = 13) | fail | mean D_b vs B1 -0.0002 [-0.0141, 0.0133]; vs B1w -0.0031 [-0.0150, 0.0082] |
| A-ii (vs B3, B4, B5) | fail | B3 0.0098 [-0.0053, 0.0276]; B4 -0.0050 [-0.0157, 0.0048]; B5 -0.0012 [-0.0157, 0.0139]; B5 eligible (r_sat < 0.9 in 100% of cells) |
| A-iii (A2 within-backbone, 13 backbones complete) | fail | vs B6 -0.0095 [-0.0151, -0.0039]; vs B5 0.0009 [-0.0004, 0.0023]; mean within-backbone Spearman -0.259 [-0.481, -0.032] |
| A-iv (ΔQ level) | fail | median zero-parameter \|ΔQ_pred/ΔQ_meas − 1\| = 0.984 over 27 Camelyon cells (bar ≤ 0.35); pooled R²_oos of LOBO-calibrated ΔQ_pred vs B0 = -1.054 (bar > 0) |

r_sat ≥ 0.9 in 0% of Camelyon cells, so the INCONCLUSIVE row does not apply. Both readings of the two ambiguous aggregations (DA-4, DA-5 in `DEVIATIONS.md`) give the same verdict (verdict_reading_dependent = False).

## A0 unit tests (all pass)

- A0a toy reproduction: pass = True. A0b Track A reproduction (3 cells, |Δ_own − Δ_R3| ≤ 1e-6): pass = True. A0c unit conversion (1e-9 relative): pass = True.
- A1 own Δ vs R3 item-1 Δ (`mahalanobis_l2`, strict seen) over all 158 cells: max |diff| = 5.3e-10.

## A1 (Camelyon LOBO over 13 backbones)

| predictor | MAE | R²_oos vs B0 | Spearman (perm p) |
|---|---|---|---|
| P | 0.0360 | 0.286 | 0.258 (0.192) |
| P_zero_param | 0.0915 | -3.367 | 0.407 (0.081) |
| B0 | 0.0390 | 0.000 | — |
| B1 | 0.0358 | 0.169 | 0.379 (0.100) |
| B1w | 0.0328 | 0.191 | 0.176 (0.276) |
| B3 | 0.0457 | -0.389 | -0.934 (1.000) |
| B4 | 0.0310 | 0.269 | 0.115 (0.349) |
| B5 | 0.0348 | 0.104 | 0.214 (0.243) |

P = P_Δ with the LOBO calibration a + b·P_Δ; P_zero_param = P_Δ with a = 0, b = 1. Family, within-family and medbench blocks: `a1_family.csv` (report only, not in the verdict). Per-backbone table: `a1_backbones.csv`; per cell × fold: `cells.csv`.

## A2 (within-backbone designs, 13 backbones × 2 folds × 30 designs × 10 repeats = 7,800 fits)

S5 sanity (G' = 15, n' = all reproduces the A1 fold Δ to 1e-6): passed for all 26 backbone × fold pairs (`a2_s5.csv`).

| predictor | pooled MAE | R²_oos vs B0 |
|---|---|---|
| P | 0.1078 | 0.001 |
| P_zero_param | 0.1381 | -0.874 |
| B0 | 0.1085 | 0.000 |
| B3 | 0.1088 | 0.003 |
| B4 | 0.1036 | 0.088 |
| B5 | 0.1087 | -0.003 |
| B6 | 0.0983 | 0.187 |

## A3 estimator audit (report only)

- 158 cells, 316 cell × fold rows. Cells with r_sat ≥ 0.9: 0; δ̂ ≥ 0.5: 0; N < 5d (formula regime unverified): 103.
- Median group-jackknife relative SE: ρ̂_w 0.057, ΔQ_pred 0.139. Median n_w/n̄ = 1.20. Largest effect of the A0c conversion factor on ΔQ_pred: 3.01%. Class-conditional centring: not evaluable (no cell has `labels_train`, DA-11).

## Required caveats

- Seen vs unseen eval sets differ also in hospital / class composition; Camelyon is stratified by hospital in the fold construction (hospitals {0, 3, 4}, 5 slides per hospital per fold); the medbench datasets may not be.
- Deviations: `DEVIATIONS.md` (DA-1 … DA-14); none changes a criterion, threshold, seed or grid.

Files: `tests.csv` (Holm within A), `cells.csv`, `a1_lobo.csv`, `a1_family.csv`, `a1_backbones.csv`, `a2_designs.csv`, `a2_lobo.csv`, `a2_s5.csv`, `a3_audit.csv`, `scatter_dq.png`, `scatter_delta.png`, `a2_dose.png`, `verdict.json`.
