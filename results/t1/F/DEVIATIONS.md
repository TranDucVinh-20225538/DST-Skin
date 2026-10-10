# Item F — deviations and recorded choices

- **DF-1 (F1 draws).** F1 re-implements bundle `s3_blind.py`, `s3b_balanced.py`, `s3c_cf.py` with the T1 replicate
  seeds of rule 10 (`SeedSequence([20261010, 6, stable_hash(config), rep])`), the same sample sizes and the same number
  of shifts as the bundle (240 / 240 / 96 for s3 at d = 8 / 32 / 128; 48 for s3c; 60 per scorer for the s3b balanced
  rows; 12 per two-block point). These are the registered F1 numbers (`f1_ref.csv`, `f1.json`).
- **DF-2 (corr(AUROC, Φ(z)) not a gate).** The order quotes "corr(AUROC, Φ(z)) ≥ .995" as part of the reference; the
  bundle's own value for Euclid at d = 8 is 0.994, so it cannot be a reproduction gate. It is reported per d × scorer
  (all T1 values ≥ 0.9956). The gates are the stated tolerances (±0.02 sign accuracy, ±0.005 balanced means).
- **DF-3 (which s3b rows are "balanced means").** Gated at ±0.005: the s3b mean AUROC of the blind scorer on its own
  balanced shifts (d = 32, 128; 6 rows) and the two-block Mahalanobis AUROC (8 rows). The two-block half-contrast AUROCs
  are reported against the bundle without a gate.
- **DF-4 (implementation check, not a gate).** After F1 failed (3 of 29 gated rows), the same code was run once with the
  bundle's seeds (`default_rng(0..n−1)`): it reproduces every bundle value to the printed precision
  (`f1_diag_bundle_seeds.csv`). This shows the three misses are Monte-Carlo differences between independent draws, not an
  implementation error. It does not change F-i, which stays `fail` (rule 18(f)); the F1 tolerances carry no Monte-Carlo
  allowance (for s3 at d = 8, the SE of a difference of two independent 240-shift accuracies near 0.97 is ≈ 0.016).
- **DF-5 (consequence).** Order F1: "Fail → the F items stop (not S3)". F2, F3 and F4 are `not run`; F-ii, F-iii, F-iv are
  `not run`. No F2/F3/F4 computation was started.
