# Multibench group-leakage check: README

Precommit: `decisions/precommit_group_leakage_multibench_2026-10-06.md` (commit 909f710), committed
before any number. Bar = 0.02 AUROC throughout. Files: `summary.md/.csv` (per arch), `A_all.csv`,
`B_all.csv` (per run), per-model JSON in `iwildcam/`, `rxrx1/`; inventory in `inventory.md`.

## Pass / fail per bar

| dataset | (A) archs with loss > 0.02: Maha / kNN / ViM / ReAct | (A) present for | (B) archs gap > 0.02: MSP / Energy (either) | (B) | (C) winner changed | median tau | verdict |
|---|---|---|---|---|---|---|---|
| MIDOG | not run (no seen-group ID split exists) | — | not run | — | — | — | absent by construction |
| iwildcam | 8 / 8 / 8 / 0 of 8 | Mahalanobis, kNN, ViM | 4 / 4 (4) of 4 | PASS | 8 of 8 | -0.05 | present |
| rxrx1 | 4 / 4 / 1 / 0 of 4 | Mahalanobis, kNN | 0 / 0 (0) of 4 | fail | 2 of 4 | 0.81 | present |

(A) bar: loss (−Δ) > 0.02 on >= half of the archs. (B) bar: gap > 0.02 (mean over seeds) on
>= 3 of 4 archs for MSP or for Energy (the per-arch "MSP or Energy" count is shown in brackets).

## Generalization verdict (locked rule)

Leakage present on 2 of 3 datasets (MIDOG counts as absent by construction) → **group-level leakage in WILDS-style ID splits (present on RxRx1 and iWildCam; MIDOG absent by construction)**.

## Caveats (descriptive, not used to change any bar)

- iwildcam: unseen-group accuracy < 0.8 (confound flag) for convnext_tiny, densenet121, resnet18, resnet50.
- rxrx1: unseen-group accuracy < 0.8 (confound flag) for convnext_tiny, densenet121, resnet18, resnet50.
- rxrx1: standard-protocol AUROCs range 0.43-0.58, i.e. detectors are near chance; leakage on RxRx1 is measured on weak detectors (1,139-class models after 10 epochs).

## Budget

Estimate (precommit): 15-22 GPU-h. Actual (sum of GPU job elapsed time, extract + base + retrain): 14.5 GPU-h.
No seed dropped (projection stayed under 40 GPU-h).

## Skipped

- MIDOG: (A)/(B) not run; standard ID split is case-disjoint from train with a single ID scanner, so
  no seen-group ID set exists (recorded in the precommit; no substitute).
- Nothing else skipped; RxRx1 was downloaded (WILDS v1.0, public) and used.

