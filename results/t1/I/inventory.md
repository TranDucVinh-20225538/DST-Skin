# Item I — inventory

Full list (path, size, mtime, sha256): `inventory_files.csv`. Cell registry: `registry.csv` (158 cells).

| category | files | location |
|---|---|---|
| R3 item-1 inputs (`<cell>.npz`, `<cell>_head.npz`) | 316 | `~/r3work/item1/inputs/` |
| R3 item-1 outputs (`paper_ci.csv`, + `results/r3/1/paper_ci_v2.csv`) | 215 | `~/r3work/item1/out_v2/*/` |
| CNN feature caches (Camelyon) | 118 | `outputs/features/camelyon17/` |
| retrain-arm code, precommit and logs (isbi_patch2, p2d) | 109 | `scripts/r3/item5_b_isbi_patch2.py`, `decisions/precommit_isbi_patch2_2026-10-06.md`, `logs/`, `~/r3work/p2d_*.log` |
| paper-3 caches (`*_z.npz`) | 33 (15 distinct in `paper-3-v2`) | `~/Downloads/` |
| MICCAI campaign scores | 22 | `outputs/rigor_pack/miccai_campaign/scores/` |
| R3 item-3 whitening caches | 13 | `outputs/rigor_pack/r3/item3/` |
| foundation-gate features (Camelyon) | 8 | `outputs/rigor_pack/foundation_gate/feats/` |
| slide_identity.csv | 1 | `outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/` |
| R3 item-6 dose.csv | 1 | `results/r3/6/` |
| Camelyon17 WILDS metadata | 1 | `data/raw/wilds/camelyon17_v1.0/metadata.csv` |

No category is missing.

## Cells per dataset and family (`registry.csv`)

| dataset | CNN | FM | total |
|---|---|---|---|
| camelyon | 12 | 15 | 27 |
| breakhis | 60 | 25 | 85 |
| dermamnist | 12 | 5 | 17 |
| isic2019 | 12 | 5 | 17 |
| kermany | 12 | 0 | 12 |
| **total** | 108 | 50 | 158 |

Camelyon backbones with `has_delta_mahalanobis_l2`: **13** (CNN: convnext_tiny, densenet121, effb3, efficientnet_v2_s,
mobilenet_v3_large, regnet_y_3_2gf, resnet18, resnet50; FM: conch_v1_5, dinov2_vitb14, dinov2_vitl14, uni, virchow2).
Camelyon cells: `n_groups_fit` 15/15 in every cell; fold sizes from the cached `fold_train`.
Kermany has no FM cell in R3 (recorded as is).
