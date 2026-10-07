# Medbench precommit: technical decisions and deviations (kept separate so the deposited sha256 stays valid)

All written after Phase 0 and before any model is trained or any AUROC / accuracy is computed.

1. REPRO check. First run (8 BLAS threads) matched kNN and the standard columns to < 1e-8 but the
   Mahalanobis 2-fold columns differed by up to 7.0e-6. The committed numbers were produced with 16 CPUs
   and no thread variables; rerun with that configuration: max abs diff 0 → PASS. No tolerance change.
2. DermaMNIST image → ISIC id mapping: `dermamnist_split_info.csv` from the DermaMNIST-C authors'
   repository (allowed by Phase 0 "Abhishek metadata if it provides the mapping"). Verified: label
   agreement 1.000; pixel check (MedMNIST centre-crops to a square before resizing; the first check
   omitted the crop and flagged S3, a check bug) 94.7% of 300 sampled images with ρ >= 0.98, median 0.994.
3. Kermany "v2" from Mendeley has train 83,484 / test 1,000 and no val/ folder (the 968 + 32 split is the
   Kaggle repackaging); used as is. (A) OOD = all v2 DRUSEN images (8,866); (B) and the M_std(v3) arm
   OOD = all v3 DRUSEN images; far OOD = the 624 v3 chest_xray test images.
4. BreakHis repeats: the 5 P_out sets are the folds of a subtype-stratified 5-fold patient partition
   (each is a random 20% of patients, as locked). Each repeat's P_out has 11-16 patients (< 20 groups);
   the BreakHis cell = (arch, seed) pooled over the 5 repeats (72 patients, 6,896 images), which is the
   unit for the sample-size floor and the bars. Phase 0 counts 81 patients (patient 14-13412 has images
   in two subtypes; the expected 82 counts it twice).
5. A-fit folds = the (B) group folds restricted to the M_std training groups (one fold assignment per
   dataset, "same 2 folds").
6. EfficientNet-B3 staging uses short side 343 = round(300 × 256 / 224), the recipe's eval Resize, instead
   of the 320 written in L6; 224-px archs use 256 as locked.
7. EfficientNetV2-S runs at 224 (L2) via the recipe transforms with input_size = 224.
8. Brain (optional): only M_std is wired so far; the cvind (B) arm is added only if the budget allows
   (L6 cut order puts brain first).

## 2026-10-07 (after training and scoring)

9. Scoring threads: training ran with OMP/OPENBLAS/MKL = 8 as locked; the CPU scoring jobs
   (`run_medbench_scores.sh`, `run_medbench_sens.sh`) ran with 16 threads. Effect is at the ~1e-6 level on
   Mahalanobis (deviation 1); no bar is that close.
10. DermaMNIST-E test as a second clean ID_unseen set for (A) was not wired into the M_std runs (their npz
    hold test_seen / test_unseen only); DermaMNIST-E is reported through the DermaMNIST-C external arm only.
11. Brain cvind (B) arm not run. Brain M_std ID_unseen = 2 images / 2 patients → underpowered (L4 floor);
    brain counts toward no bar and is not a core dataset.
12. Aggregation choices not spelled out in the precommit (gate application per run, arch = mean over passing
    runs, floors, robust bar, CI-significant swap definition, BreakHis (B) = M_std(r) gap) are listed in the
    README "Aggregation choices" section.
13. Readouts added after the first look at Kermany results (descriptive, no bar): A-gap with ID_unseen split
    into v2 test (86 images, underpowered) and v3 supplement (750 images); Δ_fit-only summary per dataset;
    ISIC 2019 / Kermany / BreakHis population-confound notes. The locked ISIC 2019 sensitivity (drop images
    without lesion_id) was run as specified.
14. M_gd(r) BreakHis runs: 17/40 fail G2/G3 on P_out (models near chance on unseen patients); reported, not
    retrained (L5).
