# Track C risk-coverage: skipped

Precommit L4 (C): risk-coverage only if `scripts/pathology_risk_coverage.py` runs on the per-image caches with
< 50 lines of wrapper. It does not: the script refits its own scorers from the full Camelyon / MIDOG feature
files at hard-coded paths (`outputs/features/camelyon17/frac1/seed42`, `outputs/features/midog/seed42`) and has
no entry point for the campaign score caches (`outputs/rigor_pack/miccai_campaign/scores/*.npz`) or the
fold-fit / leaky-vs-new-ID split. Adapting it would exceed the 50-line limit, so it is skipped as precommitted.
