# Precommit: group-leakage OOD check on medical benchmarks with leaky standard splits (medbench)

Date: 2026-10-06 (written 12:20 +07). Branch `rigor-pack`. Committed before Phase 0 and before any
number of this check is computed; a sha256 of this file is written to the DEPOSIT file for an external
deposit (OSF / Zenodo, done by the user). Not edited after results; deviations go to a dated
"Deviations" section appended at the end, with the reason.

Execution order (user, 2026-10-06 11:56): multibench → this medbench → mechanism/fix. Unchanged: every
earlier precommit and bar; Ledoit-Wolf float64 Mahalanobis; published scorer config
`OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)` with
OODScorer keys msp, energy, logit_norm (= ELogitNorm), react_energy, vim, mahalanobis, knn; `calc_auroc`;
OMP/OPENBLAS/MKL threads fixed at 8.

Known before writing: the multibench (A) readings already reported to the user (iWildCam fit-score loss
0.07-0.36 on 8/8 archs; RxRx1 standard AUROCs 0.43-0.56, i.e. near chance — the reason for the epoch
rule and gate L5), all Camelyon / patch-1 / patch-2 results, and download accessibility (below). No
number of any dataset of this check has been computed.

Access (checked 2026-10-06 ~12:00, no login used): public = ISIC 2019 (S3), ISIC 2018 Task 3
LesionGroupings (S3), MedMNIST+ npz (Zenodo 10519652), DermaMNIST-C/-E npz + CSV (Zenodo 11101338),
Kermany Mendeley rscbjbr9sj v2 (OCT2017.tar.gz 5.8 GB, ChestXRay2017.zip 1.2 GB) and v3 (ZhangLabData.zip
8.4 GB), BreakHis (UFPR direct link), CRC-VAL-HE-7K (Zenodo 1214456), Cheng figshare 1512427 (4 zips).
Not public: Br35H (Kaggle login) → brain OOD = Kermany pediatric CXR only (decided now, as instructed).
OASIS-3 is not on disk → Phase OASIS is skipped (needs a DUA; recorded, not substituted).

## L1. Datasets, groups, OOD sets (primary OOD first; secondaries reported only)

| # | dataset | group | ID classes | primary OOD | secondary OOD |
|---|---|---|---|---|---|
| 1 | DermaMNIST (MedMNIST+ 224) | HAM lesion_id | all 7 | PAD-UFES-20 (repo copy) | ISIC 2019 BCN images whose lesion is not in HAM; BloodMNIST test (far) |
| 2 | ISIC 2019 | lesion_id (no id → singleton group) | MEL, NV, BCC, AK, BKL, SCC | held-out DF + VASC | PAD-UFES-20 |
| 3 | Kermany OCT | patient (`CLASS-<pid>-<n>`) | CNV, DME, NORMAL | held-out DRUSEN | Kermany pediatric CXR (far) |
| 4 | BreakHis (all magnifications) | patient (filename) | B-A, B-F, B-TA, M-DC, M-LC, M-MC (Oh et al. OSR config 4) | B-PT + M-PC | CRC-VAL-HE-7K (far) |
| 5 opt. | Brain MRI, Cheng figshare | patient (cjdata.PID) | glioma, meningioma, pituitary | Kermany pediatric CXR | — |

ISIC 2019 sensitivity: rerun the readouts dropping images without lesion_id. OOD images / lesions never
enter any ID set (DF/VASC lesions with an ID-class image are removed from OOD and counted).

## L2. Archs, seeds, recipe

- Core archs: ResNet18, ResNet50, DenseNet121, ConvNeXt-T; seeds 42 and 43. M_std additionally trains
  MobileNetV3-L, RegNetY-3.2GF, EfficientNet-B3 @300, EfficientNetV2-S @224 at seed 42 (8 archs for the
  A-bar). Group-fold / repeat / subsample RNG: `numpy.random.default_rng(0)` (seed 0).
- Recipe: `src/models/cnn_family.recipe(backbone)` + `input_size`, ImageNet init, cosine schedule, the
  train / eval transforms of `scripts/skin_newcnn.py` (RandomResizedCrop(0.7-1), H+V flip, ColorJitter
  0.5/0.5/0.5/0.1, GaussianBlur p 0.5, RandomAffine 20°/0.1/0.8-1.2; eval Resize + CenterCrop).
  Checkpoint rule (repo rule of skin_newcnn.py, recorded): best macro one-vs-rest AUC on a validation
  set, identical across arms. Validation = the official val split where the standard split has one with
  >= 100 images (DermaMNIST M_std); otherwise (and in every (B) arm) a 10% image-level random subset of
  that arm's training images (seed 0), disjoint from every ID / OOD evaluation set. (Kermany v2 val has
  32 images → 10% of train, decided now.)
- Epochs per dataset (identical for every arm of a dataset): epochs_d = clip(ceil(300000 / n_train_std_d),
  10, 50), n_train_std_d = training images of M_std. Rationale: the multibench RxRx1 models were near
  chance after a fixed 10 epochs on 1,139 classes; a fixed 10 epochs undertrains small datasets.
  Expected ≈ DermaMNIST 43, ISIC 2019 16, Kermany 10, BreakHis 50, brain 50; the exact values from
  Phase-0 counts are written to the README before training.

## L3. Designs

M_std (standard leaky split): DermaMNIST official split; ISIC 2019 class-stratified random image 80/20
(seed 0); Kermany v2 train / test; BreakHis per repeat r = 1..5: 20% of patients held out (P_out,
stratified by subtype), M_std trained on a random 80% of the images of the remaining patients (P_in),
the other 20% of P_in images = ID_seen; brain: image-level 80/20.

(A) Frozen-model rescoring on M_std. ID_seen = ID-eval images whose group is in M_std's train set;
ID_unseen = ID-eval images whose group is not. Supplements for ID_unseen: Kermany adds v3-test images
of patients absent from v2 train; DermaMNIST adds DermaMNIST-E test (ISIC 2018 test) as a second clean
set (reported separately); BreakHis uses P_out. Same model, same OOD set. ID_seen and ID_unseen are
subsampled to identical class histograms (largest common per-class count; 20 random subsamples,
AUROCs averaged).
- A-gap = AUROC(OOD vs ID_seen) − AUROC(OOD vs ID_unseen), all 7 scores.
- A-fit: group-disjoint 2-fold scorer fit for Maha / kNN / ViM / ReAct exactly as leakfree_knn.py +
  patch 1 (fit on fold-A train groups; ID = eval images whose groups are in fold B; swap; average).
  Δ_fit = same-group-fit AUROC − disjoint-fit AUROC (positive = inflation). Folds: training groups split
  2-way balanced by group count and image count within class (greedy on shuffled groups, seed 0).

(B) Retrain on a group-disjoint split (same 2 folds). Train on fold-A groups, holding out 15% of the
images of fold-A groups with >= 2 images as ID_seen; ID_unseen = fold-B images, class-matched as in (A);
swap. Primary: within-model gap = AUROC(seen) − AUROC(unseen) for MSP / Energy / ELogitNorm; the 4
feature scores under the same protocol (fit = fold-A training images) as patch-2 P5. Secondary:
AUROC(M_std, leaky ID) − AUROC(M_gd, unseen ID).
- DermaMNIST (B) = lesion-disjoint rebuild from HAM lesion IDs; external arm (reported, not in bars):
  train on the official DermaMNIST-C split, evaluate on its test and on DermaMNIST-E test.
- Kermany (B) = rebuild on v3 train; descriptive arm (not in bars): M_std(v3) on the official v3 split,
  core archs × 2 seeds.
- BreakHis: M_std(r) gives the within-model gap; (B) adds M_gd(r) trained on all P_in images,
  evaluated on P_out.
- Brain: (B) = Cheng cvind patient folds (5 folds, folds used pairwise as in the 2-fold design:
  train on 4 folds, ID_unseen = held-out fold).

## L4. Bars and decision rules

- A-bar (dataset × score): A-gap > 0.02 or Δ_fit > 0.02 on >= half of gate-passing archs → "leakage
  present for that score on that dataset".
- B-bar: within-model gap > 0.02 on >= 2/3 of gate-passing retrained core archs (mean over seeds and
  folds) → "logit scores also benefit from group sharing".
- Uncertainty: paired cluster bootstrap (resample groups in the ID sets, images in the OOD set; same
  resample for all 7 scores), B = 2000, 95% percentile CI for every Δ / gap and every pairwise AUROC
  difference behind a rank swap. Primary bars use point estimates; a secondary "robust" bar also
  requires the CI to exclude 0. Both reported.
- Ranking: rank the 7 scores under the leaky protocol (ID_seen / standard fit) and the group-disjoint
  protocol (ID_unseen / disjoint fit) per (dataset, arch, seed); winner under each and Kendall tau-b.
  "Ranking changed" if the winner differs on >= half of gate-passing archs or median tau-b < 0.5,
  else "stable". A swap counts only if the CI of the pairwise difference excludes 0 (plain count also
  reported).
- Secondary directional prediction: median Δ of the feature family (Maha, kNN, ViM) > median Δ of the
  logit family (MSP, Energy, ELogitNorm) on >= 3/4 of core datasets tested; ReAct |Δ| <= 0.02.
- Dataset verdict: leakage present if A-bar holds for >= 1 score or B-bar holds.
- Generalization rule (combined with the multibench verdict): present on >= 3 of the 4 core medical
  datasets tested → "group leakage in standard medical splits inflates post-hoc OOD AUROC across
  benchmarks"; exactly 2 → "dataset-dependent"; <= 1 → "Camelyon/WILDS-specific; medical datasets are
  negative controls". If Phase-0 STOPs leave < 3 core datasets, the claim is capped at "dataset-dependent".
- Sample-size floor: a cell whose ID_unseen has < 200 images or < 20 groups is "underpowered" and
  counts toward no bar.

## L5. Model-quality gate (applied before interpreting any Δ)

- G1: final-epoch train accuracy >= 0.80 and final train loss <= 50% of epoch-1 loss.
- G2 on ID_seen: DermaMNIST acc >= 0.70; ISIC 2019 6-class balanced acc >= 0.50; Kermany 3-class acc
  >= 0.90; BreakHis 6-subtype balanced acc >= 0.50; brain acc >= 0.85.
- G3: a score cell with AUROC in [0.45, 0.55] under both ID variants is "near-chance" and excluded from
  bar counts (reported); >= 5/7 near-chance scores → model "inadequate (RxRx1-like)", excluded.
- G1/G2 failures are reported, excluded from bar denominators; < 3 passing archs on a dataset →
  verdict INCONCLUSIVE (model quality). Never retrain with new hyperparameters to pass a gate.
- ID accuracy (seen, unseen) next to every AUROC; unseen-group accuracy < 0.8 → confound note.

## L6. Budget and cut order

Estimate ≈ 70 GPU-h core (DermaMNIST ≈ 9, ISIC 2019 ≈ 8, Kermany ≈ 30, BreakHis ≈ 15, +15% overhead ≈ 9)
+ ≈ 3 brain. Re-projected after the smokes (1 core arch × 1 fold per dataset). If > 110 GPU-h, cut only in
this order: (1) optional brain, (2) DermaMNIST-C/-E external arm, (3) seed 43 of Kermany M_std(v3).
Never cut core archs / seeds of A or B, or BreakHis repeats. QOS <= 4 GPUs (shared with multibench and
the user's project); data pre-resized once (short side 256; 320 for EfficientNet-B3) into
`data/staged/<dataset>_<size>.tar`, copied to node-local `/tmp/$SLURM_JOB_ID/`, removed by an EXIT trap.

## Phase 0 (CPU, after this commit; committed before any training)

DermaMNIST: map MedMNIST+ 224 images to HAM ISIC ids (Abhishek metadata if it provides the mapping,
else within-class 64×64 grayscale zero-mean normalised correlation ρ >= 0.98), attach lesion_id; report
train-val / train-test / val-test lesion overlaps vs the published 886 images / 641 lesions (train-test);
verify DermaMNIST-C/-E lesion-disjointness; report-only: repo ISIC 2018 train vs val lesion overlap;
OCTMNIST / PneumoniaMNIST val/test → Kermany v3 patient overlaps (cap 6 CPU-h, else "not run").
Kermany: version fingerprint (file counts, val/ folder, sha256 of sorted filename list); patient-id
parse rate; v2 test→train patient overlap (expected ≈ 92%), v3 test→v3 train (expected 0), v3 test→v2
train; v2/v3 filename / image sharing (pixel match on 2k sample if renamed).
ISIC 2019: lesion_id coverage (expected 23,247 / 25,331; 11,847 lesions), composition by source; the
seed-0 80/20 test images with lesion in train (expected ≈ 60%); no DF/VASC lesion with an ID image.
BreakHis: parse `SOB_<B|M>_<type>-<yy>-<patient>-<mag>-<seq>.png`; patients (expected 82), images
(7,909), images per patient, patients in > 1 subtype, image-split test share with patient in train;
the 5 P_in / P_out repeats (seed 0) to CSV. Brain: PID, cvind (transpose column-major), 233 patients,
image-split leakage.
STOP rules per arm: S1 clean (< 5% of standard ID-eval images share a group with train → "clean — not
tested"); S2 groups recoverable for < 90% of images (ISIC 2019 < 85%) → STOP; S3 a Phase-0 number off by
> 25% relative from the expected value → pause, look for a mapping / parse bug, resume only if a
technical bug is fixed, else report and do not run; S4 access (above). No substitution after results.

## Order, deliverables

REPRO check (leakfree_knn.py Camelyon ResNet18 seed 42 to 1e-6, job submitted before this file) → this
precommit + DEPOSIT → Phase 0 (commit) → staging → smokes → full grid in waves → CPU analysis.
Outputs `outputs/reports/rigor_pack/medbench/`: inventory.md, phase0_split_audit.md/.csv + split CSVs,
model_quality.csv, per dataset A_gap / A_fit / B_gap / bootstrap_ci / ranking CSVs, summary.md/.csv
(with the Camelyon / multibench rows), README.md (bars, generalization verdict, directional prediction,
Kermany v2-v3 and DermaMNIST-C/-E arms, GPU-h estimate vs actual, every STOP / skip, commit hashes),
optional claim_edits.md. No home paths, data, images, tars or `.pt` committed.
