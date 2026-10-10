# T1 item I — Inventory and Camelyon17 manifest

- Precommit commit: `e326d392fd42ced26fbebe6c0a38ed983267d187`. Script `scripts/t1/item_I.py`, SLURM job 65257 (CPU, 32 cores), wall clock 542 s.
- Outputs: `registry.csv`, `inventory.md`, `inventory_files.csv`, `camelyon_manifest.csv`, `camelyon_manifest.md`, `manifest_checks.json`, `MANIFEST_OK`, `DEVIATIONS.md`.

## Counts

| dataset | CNN | FM |
|---|---|---|
| camelyon | 12 | 15 |
| breakhis | 60 | 25 |
| dermamnist | 12 | 5 |
| isic2019 | 12 | 5 |
| kermany | 12 | 0 |

Camelyon backbones with cached features and R3 item-1 Δ for `mahalanobis_l2`: **13** (≥ 8) → **S4 not triggered**.

## Camelyon17 manifest (real metadata, `load_camelyon_metadata`)

| hospital | train slides | train patches | id_val patches |
|---|---|---|---|
| 0 | 10 | 53,425 | 6,011 |
| 3 | 10 | 116,959 | 12,879 |
| 4 | 10 | 132,052 | 14,670 |

| check | result |
|---|---|
| 30 training slides, exactly 10 per hospital over {0, 3, 4} | true |
| OOD all hospital 2 (85,054 patches, 9 patients), no OOD slide / patient among training | true |
| every Camelyon cache's slide set equals the metadata's (27 cells) | true |
| every fold of every Camelyon cell holds 5 slides per hospital | true |
| totals 302,436 / 33,560 / 85,054 vs documented | equal (no WARN) |
| per-slide train patches min / max | 1,430 / 55,149 |

**MANIFEST_OK = true** → **S11 not triggered**.

GPU-hours: estimated 0, actual 0 (CPU only). Deviations: 4 (`DEVIATIONS.md`).

**Status: PASS**
