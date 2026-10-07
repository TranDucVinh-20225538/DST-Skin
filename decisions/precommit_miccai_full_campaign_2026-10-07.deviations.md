# Deviations: precommit_miccai_full_campaign_2026-10-07.md

1. (2026-10-07, before any campaign number) GPU concurrency raised from <= 3 to <= 4 on the user's instruction
   (the other project released its GPU).
2. (2026-10-07, before any campaign number) REPRO failure, diagnosis and fix. No campaign number has been read
   at any point of this item.
   - REPRO 1 (job 63855, 10:55): `leakfree_knn.py` ResNet18 / ResNet50 seed 42 vs the committed
     `outputs/reports/rigor_pack/leakage/leakfree_knn.csv`. kNN columns matched exactly; only the Mahalanobis
     2-fold columns differed (max abs diff 3.2e-5 for ResNet18, 4.7e-3 for ResNet50) -> FAIL. All pending
     campaign jobs were held; the output was moved to `outputs/rigor_pack/foundation_gate/repro_run1_63855/`.
   - REPRO 2 (job 63942, started 14:47) re-ran the same configuration. It was **cancelled at 15:03:11, before it
     wrote any number**, because the diagnosis below (known float32 Ledoit-Wolf instability,
     `decisions/addendum_numerical_stability_2026-10-05.md`) showed that a re-run of the same code could only
     reproduce the thread setting, not the committed CSV. It was not stopped because of any result.
   - Cause: `OODScorer.fit` fitted Ledoit-Wolf on the float32 cache, so the precision matrix depends on the BLAS
     thread count. The REPRO jobs inherited `OMP_NUM_THREADS=4` from the submitting shell; the thread setting
     of the job that wrote the committed CSV (62932) is not recorded.
   - Diagnosis (job 63944, `scripts/rigor/repro_lw_diag.py`, ResNet50 seed 42 fold 0; result in
     `outputs/reports/rigor_pack/miccai_campaign/repro_diag/`): float32 fits are identical for a fixed thread
     count but differ between 1, 4 and 16 threads (precision max abs diff ~1e8 vs the float64 reference,
     fold-0 disjoint AUROC shifted by ~1.5e-3); float64 fits are thread-invariant (relative d² diff <= 2.3e-11,
     AUROC diff 0).
   - Fix (technical, commit `fix:` 4c316f8): Ledoit-Wolf is fitted on float64 L2-normalised features in
     `OODScorer.fit`; the REPRO runner pins OMP / OPENBLAS / MKL to 16 threads. This is the precommit's own
     rule ("Ledoit-Wolf float64 Mahalanobis; float64 features in every scorer fit; thread pin"). The campaign
     scorers (`fm_gate_score.py`, `medbench_scores.py`) already pass float64 features with 16 pinned threads,
     so the fix does not change any campaign output and no campaign score is recomputed.
   - New REPRO: the H11c baseline is regenerated with the fixed code into
     `outputs/reports/rigor_pack/leakage_lw64/` (the old committed CSV is kept unchanged), and REPRO is run twice
     (`repro_lw64_a`, `repro_lw64_b`). PASS = both runs equal each other and the new baseline to 1e-6 on every
     float column of ResNet18 / ResNet50 seed 42. Campaign numbers are read only after PASS.
3. (2026-10-07) Scheduling only: Camelyon scoring tasks 63885_0, 63885_1, 63885_2, 63882_1, 63883_0, 63902_0 and
   63902_1 were requeued (restarted from scratch) to free CPUs for the REPRO and diagnosis jobs under the
   96-CPU QOS limit.
4. (2026-10-07 16:18) New REPRO **PASS**: baseline (job 63950), REPRO a (63951) and REPRO b (63952) are identical
   on every float column of ResNet18 / ResNet50 seed 42 (max abs diff 0). The comparison was run on the login
   node with the exact script of `run_fm_repro.sh MODE=compare` (job 63953 was blocked by the CPU QOS and
   cancelled); output in `outputs/reports/rigor_pack/leakage_lw64/repro_compare.txt`. The hold on campaign
   numbers is lifted.
5. (2026-10-07 ~17:00, **post-hoc, after the preliminary Track A / C / D read; descriptive only, no bar, does not
   change L7**) Requested by the user:
   - Track D two-channel table (`scripts/rigor/camp_trackD_channels.py`): within-model seen − unseen AUROC for all
     7 scores on the existing isbi_patch2 retrained models (float64 scorer), folds reported separately because
     fold 1 has unseen-slide accuracy < 0.8. Note: Track D's "published − retrain" column reads
     `isbi_patch/phaseB_bar.csv` (patch 1, seed 42 only, unbalanced folds) as locked in L4; the retrain arm
     itself (isbi_patch2) has seeds 42-44.
   - Kermany Track C confound check (`scripts/rigor/camp_kermany_confound.py`, job 63970): new-patient ID split
     into v2 test_unseen (86) and v3 supplement (750); accuracy, class histograms, class-matched TPR.
   - Ranking-flip robustness: counts of `swap_ci_significant` rows (already in `medbench_report.ranking`).
