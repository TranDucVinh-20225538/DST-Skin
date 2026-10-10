# Item I — DEVIATIONS

| id | what | why | effect on criterion | who decided |
|---|---|---|---|---|
| DI-1 | File inventory with path, size, mtime, sha256 is in `inventory_files.csv` (837 files, 11 categories); `inventory.md` summarises it. | The order asks for both; the full list is too long for markdown. | none | agent |
| DI-2 | `paper_ci_csv` points to `~/r3work/item1/out_v2/<cell>_maha/paper_ci.csv` and `_knnmc/paper_ci.csv`; the Δ flags are read from `results/r3/1/paper_ci_v2.csv`, the R3 item-1 consolidation of those files. | This is R3 item 1's final output layout. | none | agent |
| DI-3 | Paper-3 caches: 15 files in `~/Downloads/paper-3-v2/e2_distances/` (runA_grl, runB_orth1, runB × seeds 42/52/62/72/82), plus copies in `paper-3-v3/` and 3 loose files in `~/Downloads/`; all are listed. The order expected seeds such as s62 among 15 files; the v2 set is the full 15. | Location not given by the order. | none (F2 input) | agent |
| DI-4 | Item I was run in parallel with the end of item 0 (after the precommit was pushed). | Both CPU-light; rule 1 is satisfied. | none | agent |
