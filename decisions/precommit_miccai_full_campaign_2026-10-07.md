# Precommit: MICCAI full campaign (one pass, parallel tracks)

Date: 2026-10-07 (written ~11:30 +07). Branch `rigor-pack`, code HEAD at writing faf8005. Committed before
any campaign number is read: no FM feature, probe, score, AUROC, Δ or τ of this campaign has been computed.
Not edited after results; deviations go to `precommit_miccai_full_campaign_2026-10-07.deviations.md`.

Supersedes for execution (not for its locks) `precommit_foundation_leakage_gate_2026-10-07.md`: its Camelyon
FM protocol (L2 FMs 1-4, L3 Camelyon protocol, L4 gate) is reused unchanged and becomes Track A seed 42; its
"no seed sweep" STOP is lifted by the user's one-pass authorisation (recorded in its deviations file). The gate
is still computed and reported (`outputs/reports/rigor_pack/foundation_gate/`).

Known before writing (already reported, used in the bars below as existing cells): Camelyon17 CNN Δ_fit
(rigor-pack / mechanism cells, ~0.07-0.22 feature, ReAct ≈ 0); medbench CNN A-fit Δ_fit on feature scores
~0.02-0.04 (DermaMNIST, ISIC, Kermany) and 0.11-0.19 (BreakHis), ReAct ≈ 0; medbench ranking outputs; mechanism
F2 works on 2/3 (Camelyon, iWildCam, RxRx1) datasets; isbi_patch / isbi_patch2 slide-disjoint retrain
(ResNet50 / ConvNeXt-T / DenseNet121, seeds 42-44, MSP / Energy published − disjoint ≈ 0.24-0.28 at seed 42).
Inventory facts (no numbers): HF access — DINOv2-B/L (torch.hub, cached / downloading), UNI v1 cached, CONCH v1
gated 403, CONCH v1.5 (TITAN) cached, Virchow v1 gated 403, Virchow2 accessible, UNI2-h accessible.
No chest X-ray dataset on the HPC; TCGA only as RNA-seq in another user's directory; MIDOG / OpenMIBOOD crops
on disk in per-case directories.

Unchanged: every earlier precommit and bar; published scorer config
`OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)`; `calc_auroc`
(ID = higher score); Ledoit-Wolf float64 Mahalanobis; float64 features in every scorer fit; thread pin
(GPU extraction OMP / OPENBLAS / MKL = torch threads = 8; CPU scoring = 16).

Claim lock (not expanded after numbers): group leakage in ID splits inflates feature-score OOD AUROC
(Mahalanobis / kNN / ViM); ReAct stays ~flat; ranking can flip.

Resources: QOS <= 4 GPUs per user; by the user's instruction this campaign runs at most 3 GPU jobs at a time
(1 GPU kept for the user's `pragmatic` jobs). Partition defq, `--gres=gpu:1 -c 8 --mem=48G`, node002 excluded,
node-local `/tmp/$SLURM_JOB_ID` staging with EXIT-trap cleanup, never /dev/shm.

## L1. Datasets

| priority | dataset | group | tracks | locked arms / splits |
|---|---|---|---|---|
| must | Camelyon17 (WILDS v1.0) | slide | A, C, D, E | rigor-pack index order; H11c slide-disjoint 2-fold (`leakfree_fit_scores.folds(seed)`); OOD = hospital 2 |
| must | BreakHis | patient | A, C, E | medbench arms std_r0-4 (A-fit, A-gap) and gd_r0-4; OOD as medbench (non-benign/malignant subtypes + CRC-VAL) |
| must | ISIC 2019 | lesion_id | A, C, E | medbench arms std, b0, b1; OOD as medbench |
| cheap | DermaMNIST | lesion (HAM) | A, C, E | medbench arms std, b0, b1 (staged tar present → run); OOD PAD-UFES-20 primary |
| existing | Kermany | patient | C, E, L7 | existing medbench CNN caches only (no FM) |
| Track B | chest X-ray with patient ID / TCGA site | — | B | **STOP** (decided now): no CXR corpus on the HPC; TCGA present only as RNA-seq (not images) in another user's directory. No download in this campaign |
| skip | RxRx1 | — | — | never counts as medical |
| optional F | MIDOG (OpenMIBOOD) | case (per-case directory) | F | audit only, if per-image case IDs are recoverable for >= 90% of the images of the existing MIDOG caches; never in the L7 denominator |

STOP per dataset: missing data / download > 100 GB / licence click / DUA / login → skip, record, continue. No
dataset substituted after results. Nothing registered or accepted on the user's behalf. No invented IDs.

## L2. Backbones (Track A)

CNN anchors: Camelyon ResNet50 + ConvNeXt-T. Seed 42 = existing mechanism cells (re-scored by the campaign
scorer for the bootstrap CI and score caches); seeds 43, 44 = the existing seed-43 / 44 Camelyon models,
re-extracted in index order with the existing `extract_features_indexed.py` (no training). Medical CNN anchors
= all medbench archs at seed 42 (existing caches).

Foundation models, frozen, float32, no autocast, attempted in this order (load / OOM / import failure → STOP
reason, grid continues):
1. DINOv2 ViT-B/14, `x_norm_clstoken`, 768-d, Resize 224 bicubic + CenterCrop 224 + ImageNet norm.
2. UNI v1 (`hf-hub:MahmoodLab/uni`, CLS, 1024-d, same transform). UNI2-h not run (UNI v1 loads; one UNI only,
   as the foundation-gate precommit).
3. CONCH v1.5 (TITAN `build_conch`, attentional-pool contrast embedding 768-d, its own 448 transform); CONCH v1
   is gated (403) → STOP v1.
4. Virchow2 (`hf-hub:paige-ai/Virchow2`, timm, `mlp_layer=SwiGLUPacked, act_layer=SiLU`; embedding = concat(CLS,
   mean of patch tokens after the 4 register tokens), 2560-d, Resize 224 bicubic + CenterCrop 224 + ImageNet
   norm, as the model card). Virchow v1 gated (403) → STOP.
5. Optional DINOv2 ViT-L/14 (`x_norm_clstoken`, 1024-d), subject to L6.

Probe: `fm_ood_pilot.fit_linear_head` (`LogisticRegression(C=1.0, max_iter=2000, lbfgs)`, raw embeddings;
binary logits [−z, z]); per Camelyon fold / per medbench arm on that arm's training images.

Seeds (Camelyon, queued with seed 42, no waiting): s ∈ {42, 43, 44}. FM cells: features reused; seed s changes
the H11c fold RNG (`folds(..., seed=s)`) (probe lbfgs and scorer are deterministic). CNN anchor cells: the
seed-s model and the seed-s folds. Medical datasets: seed 42 only.

## L3. Protocol

Camelyon, per backbone × seed: exactly the foundation-gate L3 (standard 7 AUROCs; A-fit 2-fold Δ_fit for
Maha / kNN / ViM / ReAct with the full probe; FM logit within-model gap with the probe refitted per fold;
cluster bootstrap B = 2000 `default_rng(2)` over id_val slides and OOD images; near-chance [0.45, 0.55] under
both same and disjoint → excluded from denominators, reported). CNN anchors: the published fc in place of the
probe; the CNN logit contrast comes from Track D. Per-image score caches saved for Track C.
Medical (BreakHis / ISIC / DermaMNIST), per FM: probe per arm; medbench run npz; `medbench_scores.py`
unchanged (class-matched A-gap, A-fit on std arms, cluster bootstrap B = 2000, B within-model gap); probe
training / seen / unseen accuracy reported; unseen-group accuracy < 0.8 → confound note.

## L4. Tracks

- A: as L1-L3. No stain.
- B: STOP (L1). `trackB_new_medical/STOP_reason.md`.
- C (CPU, on caches): τ = the ID score at 95% TPR on the leaky ID set (ID = positive, score >= τ kept as
  ID; τ = 5th percentile of leaky-ID scores). Leaky / new-ID sets: Camelyon — fold-f-fit scorer, same-slide
  id_val (leaky) vs other-fold id_val (new slides), fold mean; medbench std arms — full fit, test_seen (leaky)
  vs test_unseen (+ unseen_extra for Kermany) (new groups); BreakHis std_r0-4 mean. Reported per score:
  realized TPR on new-ID = P(s >= τ), false-alarm rate on new-ID = 1 − realized TPR (new-group ID images
  flagged OOD), OOD pass rate P(s_OOD >= τ) (= FPR, same τ). "Material" (descriptive) = realized TPR <= 0.90.
  Risk-coverage: only if `scripts/pathology_risk_coverage.py` runs on the caches with < 50 lines of wrapper;
  else skipped with reason. Cells: every Track A Camelyon cell and medical FM cell, and the medbench CNN seed-42
  std caches of Derma / ISIC / BreakHis / Kermany.
- D: no new retrain (isbi_patch2 covers ResNet50 / ConvNeXt-T / DenseNet121 × seeds 42-44 × 2 folds). Table:
  (a) scorer-fit feature Δ_fit (Track A CNN anchors / mechanism cells) beside (b) backbone leakage = isbi_patch2
  within-model seen − unseen MSP / Energy (m4logit-style) and published − slide-disjoint-retrain MSP / Energy
  (isbi_patch phaseB_bar.csv), cited by path and commit.
- E (CPU, existing caches): F2 = cross-fitted scorer (each ID image scored by the fold fit not containing its
  group, OOD scores averaged over the two fits) vs F1 = disjoint 2-fold AUROC, 4 fit scores, per arch; medbench
  std arms (seed 42, all archs; BreakHis std_r0-4 mean) with `medbench_scores.fold_map` folds, ID = test_seen;
  Camelyon from the mechanism cells (+ FM cells). Per dataset: works if median |F2 − F1| <= 0.02 and median
  Kendall τ-b (F2 vs F1 over the 7 scores, logit scores unchanged) >= 0.8. "F2 works" if >= 2/3 of the medical
  datasets tested work. RxRx1 / iWildCam reported separately, never counted.
- F: eligibility check first (L1); if eligible, group-shared vs group-clean scoring on the existing MIDOG caches
  only, exploratory.
- G: integrity (this file, thread pin, LW float64, NA cells, commit hashes).
- H: outputs + L7 decision.
- I: not started in this pass (L6 cut 1, 1 GPU kept for the user's jobs; mechanism M1 already re-extracted
  Camelyon under Macenko / grayscale). `trackI_stain/followup_stain_ablation_note.md` only.

## L5. Quality

Trained CNNs: medbench G1-G3 as reported. Frozen FM + probe: probe accuracy seen / unseen reported;
near-chance cells excluded from bar denominators.

## L6. Budget and cut order

Estimate (A100): Camelyon FM extract 4-5 FMs 4-9 GPU-h; medical FM extract 1-3; CNN anchor indexed
re-extraction (2 archs × 2 seeds) 1-3; B 0 (STOP); D 0; C / E / F CPU; I 0. Total ≈ 6-15 GPU-h (+15%).
Ceiling 70 GPU-h. If the post-smoke projection exceeds it, cut only in order: Track I, Track F, DermaMNIST in
Track A, DINOv2-L / Virchow2 if not started, Track B retrain arm. Never cut Camelyon Track A seed 42 + 3-seed
anchors, BreakHis / ISIC FM seed 42, Track D inclusion, Track E on caches, Track C τ@95%, this precommit, the
decision table.

## L7. MICCAI vs MIDL decision rule (verbatim from the campaign prompt)

Denote medical datasets = Camelyon17, BreakHis, ISIC 2019, DermaMNIST (if run), Kermany (if F2/A-fit run),
Track B CXR/TCGA (if run). Exclude RxRx1. Exclude OpenMIBOOD from this denominator by default.

MICCAI framing if both:

Δ ≥ 0.05: median feature-family Δ_fit (or leaky−disjoint gap if Δ_fit unavailable) ≥ 0.05 on ≥ 2 medical
datasets (seed-42 primary; 3-seed Camelyon must not reverse the Camelyon sign — if 3-seed median feature Δ <
0.02 while seed-42 ≥ 0.05, mark Camelyon unstable and do not count it toward the ≥2).
Ranking flip on ≥ 1 medical: winner under leaky protocol ≠ winner under group-disjoint protocol on ≥ half of
gate-passing archs/backbones for that dataset, or median Kendall τ-b < 0.5 (same spirit as medbench).
Else → MIDL framing (keep medical honesty; do not claim MICCAI-scale breadth).

Secondary (descriptive, not for venue flip): ReAct median |Δ| ≤ 0.02 on Camelyon FM cells; clinical τ@95% shows
material TPR drop or FA rise on new-ID.

One line: Δ_fit (feature) ≥ 0.05 on ≥ 2 medical datasets AND ranking flip on ≥ 1 medical → MICCAI; else MIDL.

Operationalisation (fixed now):
- Per backbone: median of the non-near-chance Mahalanobis / kNN / ViM Δ_fit (seed 42; BreakHis = mean over
  std_r0-4 first). Per dataset: median over all gate-passing backbones with an A-fit cell (medbench CNN archs at
  seed 42 that pass medbench G1-G3 + loaded FMs; Camelyon: the 8 mechanism-cell CNNs + loaded FMs). CNN-only
  and FM-only medians reported beside it (descriptive).
- Camelyon 3-seed check: median over (backbone ∈ CNN anchors + FMs, seed ∈ 42-44) of the per-backbone median
  feature Δ_fit.
- Ranking flip: medbench — `medbench_report.ranking` rows (A and B families, passing runs, seed 42) of that
  dataset, CNN + FM; Camelyon — winner over the 7 scores (near-chance removed) under same_2fold (leaky) vs
  disjoint_2fold (group-disjoint), per backbone at seed 42, Kendall τ-b over the same scores. Flip if winner
  changed for >= half of the rows or the median τ-b < 0.5.

## L8. Reporting

NA + STOP reason for every empty cell; no model dropped after its numbers (skips only for load / OOM / data
STOP decided before). Commit hashes of this file, of the job code HEAD (logged in every job) and of the results.
No edits to manuscript/ or older precommits (deviations files only). No home paths, data, tars, feature arrays or
weights committed. Outputs `outputs/reports/rigor_pack/miccai_campaign/` as the campaign prompt lists.
