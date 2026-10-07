# Mechanism / predictor / fix: README

Precommit `decisions/precommit_leak_mechanism_fix_2026-10-06.md` (commit 57d8035). Files: mechanism.md/.csv,
m1_cells.csv, predictor.md/.csv, fixes.md/.csv, checklist.md, per-cell JSON in `cells/`.

## Pass / fail per bar

| item | result |
|---|---|
| M1 camelyon | group appearance is the main driver (gray ID acc 0.996 → 0.556; macenko ID acc 0.996 → 0.601; ablation destroys accuracy → reading weakened) |
| M1 iwildcam | not explained by colour / stain (colour ID acc 0.765 → 0.716; gray ID acc 0.765 → 0.589; ablation destroys accuracy → reading weakened) |
| M1 rxrx1 | group appearance is the main driver (colour ID acc 0.187 → 0.004; gray ID acc 0.187 → 0.005; ablation destroys accuracy → reading weakened) |
| M4 camelyon | fit group-level effect (R 0.93); logit group-level effect (R 0.97) |
| M4 iwildcam | fit group-level effect (R 0.58); logit group-level effect (R 0.58) |
| M4 rxrx1 | fit group-level effect (R 0.61); logit group-level effect (R 1.17) |
| P M2_balanced_acc | not predictive (ρ +0.36, LODO MAE 0.073) |
| P M3_same_group_nn_frac | not predictive (ρ +0.24, LODO MAE 0.081) |
| P centroid_distance | not predictive (ρ +0.27, LODO MAE 0.161) |
| F F2 | works (2/3) |
| F F3a | does not work (0/3) |
| F F3b | does not work (0/3) |
| F F3c | does not work (0/3) |
| F F4 | does not work (1/3) |

## Budget

Estimate: ~5-6 GPU-h (M1 re-extraction). Actual: 4.9 GPU-h.

## Technical choices / skipped

- M1 reduction median excludes cells with |original effect| < 0.01 (ratio undefined; counted per row); the all-cells median is shown too.
- M1 within-model gap: Camelyon retrained ISBI-patch-2 v2 seed-42 models exist for ResNet50 / DenseNet121 / ConvNeXt-T only (no ResNet18).
- MIDOG not used (precommit). Medbench datasets are not part of this precommit.

