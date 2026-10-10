# Item 0 — DEVIATIONS

| id | what | why | effect on criterion | who decided |
|---|---|---|---|---|
| D0-1 | `git status` was not clean at start (4 modified files under `outputs/reports/camelyon17/frac1/seed4{3..6}/pilot_summary.csv`, ~1,600 untracked logs). They were left untouched; only T1 files are staged. | Files belong to earlier work; rule 13 forbids touching `outputs/`. | none | user (2026-10-11), agent applied |
| D0-2 | Branch `t1` created from `rigor-pack` HEAD `4857c485fef134ba443131375da4b329d98297e7`; only `t1` is pushed. | Order item 0 step 1. | none | user + order |
| D0-3 | The reused `~/handoff/crossfit-ood_v3/` is **not** identical to the v3 zip: R3 added the Track A scorers (`mahalanobis_l2`, `knn_mean_cosine`), `fold_ids_eval`, and 2 test files. pytest: pristine zip (unzipped to `/tmp/t1_cfz_check`) **44 passed / 44**; reused copy 47 passed, 3 skipped, 0 failed. T1 uses the reused copy read-only (it carries the Track A scorers that rule 9 requires) with venv `~/.venvs/crossfit-r3` (`pip show` 0.0.3.dev0). | The order's 44/44 refers to the v3 package; the R3 copy is the one R3 item 1 used. | S2 not triggered (44/44 on v3) | agent (least invasive) |
| D0-4 | The precommit (step 7) was committed and pushed **before** steps 2–6. | User instruction; rule 1 allows it (stricter). | none | user |
| D0-5 | GPU concurrency limit raised from 2 (rule 14) to **4**. | User decision 2026-10-11 01:41. QOS limit is also 4. Never pre-empt others still holds. | none (execution only) | user |
| D0-6 | Microbenchmark on one GPU type only (A100-SXM4-80GB, the only type in `defq`; node001 drained, node002 down). kNN equality run on 2 cells × k ∈ {1, 50} × both kNN forms (`knn_mean_cosine`, k-th distance `knn`): a superset of "2 cells (k = 1 and k = 50)". | Only one GPU type available. | none | agent |
| D0-7 | `stable_hash` (undefined in the order) fixed in the precommit: first 4 bytes of sha256 of the string / canonical JSON. Work lists that need the registry or sampled designs are recorded in precommit addenda before their item launches. | Order does not define it / user decision. | none | user + agent |
| D0-8 | Microbenchmark kNN: float64 brute force with top-200 (k_max of item B), 50,000 × 100,000 × 2,560. | Matches the B design. | none | agent |
