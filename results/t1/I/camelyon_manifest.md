# Camelyon17 manifest (item I)

Source: `data/raw/wilds/camelyon17_v1.0/metadata.csv` via `scripts/rigor/common.py:load_camelyon_metadata` (R3 helper used by `scripts/r3/item6_dose.py`).

| hospital | train slides | train patches | id_val patches |
|---|---|---|---|
| 0 | 10 | 53425 | 6011 |
| 3 | 10 | 116959 | 12879 |
| 4 | 10 | 132052 | 14670 |

OOD (`test`) split: hospitals [2], 85054 patches, 9 patients; OOD slides among training slides: none; OOD patients among training patients: none.
id_val hospitals: [0, 3, 4]; id_val slides subset of the 30 training slides: True.
Totals train / id_val / OOD: 302436 / 33560 / 85054 (documented 302,436 / 33,560 / 85,054).
Per-slide train patches min / max: 1430 / 55149.

## paper_2fold slides per hospital per fold (every Camelyon cell)

| cell | fold 0 (h0/h3/h4/other) | fold 1 (h0/h3/h4/other) |
|---|---|---|
| camelyon_conch_v1_5_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_conch_v1_5_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_conch_v1_5_s44 | 5/5/5/0 | 5/5/5/0 |
| camelyon_convnext_tiny_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_convnext_tiny_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_convnext_tiny_s44 | 5/5/5/0 | 5/5/5/0 |
| camelyon_densenet121_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_dinov2_vitb14_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_dinov2_vitb14_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_dinov2_vitb14_s44 | 5/5/5/0 | 5/5/5/0 |
| camelyon_dinov2_vitl14_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_dinov2_vitl14_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_dinov2_vitl14_s44 | 5/5/5/0 | 5/5/5/0 |
| camelyon_effb3_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_efficientnet_v2_s_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_mobilenet_v3_large_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_regnet_y_3_2gf_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_resnet18_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_resnet50_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_resnet50_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_resnet50_s44 | 5/5/5/0 | 5/5/5/0 |
| camelyon_uni_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_uni_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_uni_s44 | 5/5/5/0 | 5/5/5/0 |
| camelyon_virchow2_s42 | 5/5/5/0 | 5/5/5/0 |
| camelyon_virchow2_s43 | 5/5/5/0 | 5/5/5/0 |
| camelyon_virchow2_s44 | 5/5/5/0 | 5/5/5/0 |

## Checks

- 30 training slides, exactly 10 per hospital over {0, 3, 4}: **True**
- OOD all hospital 2, no OOD slide or patient among the training slides: **True**
- every Camelyon cache's slide set equals the metadata's: **True** (27 cells)
- every fold of every Camelyon cell holds exactly 5 slides per hospital: **True**
- **MANIFEST_OK = true**

## Warnings (not stops)

- none
