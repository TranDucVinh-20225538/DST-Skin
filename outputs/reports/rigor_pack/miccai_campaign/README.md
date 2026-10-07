# MICCAI full campaign (2026-10-07) — README

Precommit: `decisions/precommit_miccai_full_campaign_2026-10-07.md` (f438005), DEPOSIT e5a81dc
(sha256 8755770c52828fc617da1fcf2f386ef0d1b83f302e053cd4785d59a173b41f00), committed before any campaign number.
Deviations: `decisions/precommit_miccai_full_campaign_2026-10-07.deviations.md` (items 1-6) and
`decisions/precommit_foundation_leakage_gate_2026-10-07.deviations.md`.
Results: `summary.md` (tables), `inventory.md` (files), `decision_miccai_vs_midl.md` (L7 readout, internal),
`venue_note.md` (chosen venue: MIDL 2027).

## Compute

| item | estimate (precommit L6) | actual |
|---|---|---|
| GPU (A100): FM extraction Camelyon + medical, CNN-anchor indexed re-extraction, smoke | 6-15 GPU-h (+15%), ceiling 70 | 11.4 GPU-h (27 `fm-gate-x` jobs, sacct ElapsedRaw × GPUs) |
| CPU: scoring, REPRO, diagnosis, Tracks C / E, post-hoc checks | not budgeted | ~552 core-h |

Parallelisation: GPU extraction ran as one array, at most 3 GPUs (1 kept for the user's other project) and at most
4 after that project released its GPU (deviation 1). CPU scoring ran as arrays of 16-CPU tasks under the
96-CPU-per-user QOS (at most 6 concurrent); array throttles were raised from 1 to 3 once the REPRO jobs were
running. Several scoring tasks were requeued to free CPUs for REPRO and diagnosis (deviation 3).

## REPRO

REPRO 1 (job 63855) failed on the Mahalanobis 2-fold columns. Cause: Ledoit-Wolf fitted on float32 features, so the
precision depends on the BLAS thread count (diagnosis job 63944). Fixed in `fix:` 4c316f8 (Ledoit-Wolf in float64,
BLAS pinned). New REPRO PASS (fa5ea57): baseline and two runs identical (max abs diff 0). Campaign scorers already
used float64 with pinned threads, so no campaign score changed. The H11c table regenerated with the fixed code is
`outputs/reports/rigor_pack/leakage_lw64/leakfree_knn.csv`; vs the old float32 table, |Δ Maha 2-fold| changes by
0.0067 (ResNet50), 0.0011 (EffB3), <= 0.0002 elsewhere; kNN unchanged (`leakage_lw64/f32_vs_f64.csv`).

## Skips and STOPs

- Track B (new medical datasets): STOP, `trackB_new_medical/STOP_reason.md`.
- Track F (OpenMIBOOD / MIDOG): skipped, case IDs not recoverable, `trackF_openmibood/skip_reason.md`.
- Track I (stain): not started (L6 cut 1), `trackI_stain/followup_stain_ablation_note.md`.
- RxRx1: skipped (not a medical pass, L1).
- CONCH v1: GatedRepoError 403 → precommitted fallback CONCH v1.5. UNI2-h: not run (L2).
- Kermany: CNN caches only, no FM (L1).
- Track C risk-coverage: skipped, wrapper > 50 lines (`trackC_clinical_tau/risk_coverage_skip.md`).
- No model dropped after its numbers. NA cells: Track D published − retrain at seeds 43 / 44 (the locked source,
  `isbi_patch/phaseB_bar.csv`, is seed 42 only).

## Post-hoc (labelled in their files, descriptive, not decision criteria)

Track D two-channel table, Kermany Track C confound check, slide-identity mechanism check, L4 variant with Virchow2
(deviation 5-6).

## Commits

f438005, e5a81dc (precommit + DEPOSIT); 71015a6, a5dc3e9, a99b683, 5608887 (campaign code / notes); e38e3ea
(quarantine script, unused); 92e6015 (diagnosis); 4c316f8 (fix); d2a4239 (deviations); fa5ea57 (REPRO PASS);
3e85bb0, 5a965f1 (post-hoc); 5abb301 (results); final report commit: see `git log` for this file.
Every job log records the code HEAD it ran on (`logs/`, not committed).
