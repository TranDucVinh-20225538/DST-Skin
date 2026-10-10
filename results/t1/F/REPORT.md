# Item F — blindness lemmas on real frozen features (REPORT)

**Verdict: INCONCLUSIVE** (rule 18: F-i failed and F-iv is `not run`, so neither "dropped" nor "lemma-level" is awarded;
INCONCLUSIVE is never counted as a pass).

| sub-criterion | status | numbers |
|---|---|---|
| F-i (F1 reference reproduced within tolerance) | **fail** | 26 of 29 gated rows within tolerance; misses: s3 sign accuracy d = 8 Mahalanobis 0.983 vs 0.963 (tol ±0.02), s3 d = 32 ViM-residual 1.000 vs 0.979 (±0.02), s3c Cornish-Fisher predicted mean d = 8 0.4920 vs 0.486 (±0.005) |
| F-ii (sign law, real cells) | not run | F1 failed → F items stop (order F1) |
| F-iii (Edgeworth + balanced, F3) | not run | idem |
| F-iv (blind-but-detectable, F3) | not run | idem |

- Commit at report time: see the git log of branch `t1` (this file's commit); precommit commit `e326d392fd42ced26fbebe6c0a38ed983267d187`.
- GPU-hours: 0 (F1 is CPU; F4 not run). Estimate 6–12, cap 20.
- Wall clock: F1 ≈ 1.5 min on 12 CPU processes (login node), 2026-10-11.

## F1 results (`f1_ref.csv`, `f1.json`)

Sign accuracy of the first-order law, T1 draws (bundle value in brackets): d = 8 — Mahalanobis 0.983 (0.963), Euclid 0.988
(0.975), ViM-residual 0.979 (0.983); d = 32 — 0.979 (0.988), 0.988 (0.983), 1.000 (0.979); d = 128 — 1.000 (0.990),
0.990 (0.979), 1.000 (1.000). corr(AUROC, Φ(z)) ≥ 0.9956 in every d × scorer.

s3c trace-balanced shifts (Mahalanobis): mean AUROC 0.4937 / 0.4889 / 0.4946 at d = 8 / 32 / 128 (bundle .489 / .491 /
.495; all within ±0.005); Cornish-Fisher predicted mean 0.4920 / 0.4872 / 0.4938 (bundle .486 / .490 / .495; d = 8
outside ±0.005). Sign agreement of CF with the measured AUROC − ½: 0.75 / 0.81 / 0.77.

s3b: blind-scorer AUROC on its own balanced shifts and the two-block Mahalanobis AUROC are within ±0.005 in all 14 rows;
half-contrast AUROCs match the bundle to ≤ 0.002 (report only).

**Implementation check (not a gate, DF-4):** the same code with the bundle's seeds reproduces every bundle number to the
printed precision (`f1_diag_bundle_seeds.csv`: e.g. 0.9625 vs 0.963, 0.4892 vs 0.489, CF 0.4861 vs 0.486). The three misses
are differences between independent Monte-Carlo draws; the registered F1 tolerances have no Monte-Carlo allowance.
F-i stays `fail` as registered.

## Required caveats

- The F-i failure is a reproduction-tolerance failure on independent draws, not evidence against the sign law; it is
  reported as registered and it stops F2–F4 by the order's rule. Whether to amend the F1 tolerance is a decision for the
  order's owner; nothing here pre-empts it.
- Deviations: `DEVIATIONS.md` (DF-1 … DF-5); none changes a criterion, threshold, seed or grid.

Files: `f1_ref.csv`, `f1.json`, `f1_diag_bundle_seeds.csv`, `tests.csv`, `verdict.json`.
