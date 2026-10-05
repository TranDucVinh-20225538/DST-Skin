# Multibench group-leakage inventory (2026-10-06, before any score)

| dataset | on HPC | models | group IDs | standard ID shares groups with train? | plan |
|---|---|---|---|---|---|
| Camelyon17-WILDS | yes | 8 archs × 5 seeds | slide, hospital (indexed caches `rigor_indexed/`) | yes (id_val = same 30 slides) | reference (H11c, ISBI patches 1-2) |
| MIDOG (OpenMIBOOD 1a/1b/1c crops, 107 MB) | yes | 8 archs × seeds 42-46 (+ OpenMIBOOD R50) | case = folder `1a/<case>/` in imglists | no: train 40 cases, valid 5 (020, 026, 031, 032, 033), test 1a 5 (003, 017, 025, 027, 049), pairwise disjoint; one ID scanner (1a) | not run: no seen-group ID under this protocol; "no group sharing by construction" |
| iWildCam-WILDS v2.0 (12 GB) | yes | 8 archs, seed 42 (`data/models/iwildcam/seed42/`) | `location_remapped` in metadata.csv | yes: train 129,809 images / 243 locations; id_val 7,314 / 146 locations, all in train; OOD test 42,791 / 48 locations, none in train | (A) 8 archs after indexed re-extraction (existing caches used a shuffled train loader, no index); (B) 4 archs × 2 folds × seeds 42/43 |
| RxRx1-WILDS v1.0 (7.4 GB compressed) | downloading (direct WILDS URL, no licence click) | none | experiment / plate / well / site in metadata | yes by WILDS design: `id_test` = site 2 of the training wells | train 4 base models (seed 42) for (A); (B) 4 archs × 2 folds × seeds 42/43 |

Scanner grouping on MIDOG: train and ID are both scanner 1a and OOD are the other scanners, so a
scanner-disjoint ID set cannot be formed without changing the benchmark; not substituted.
