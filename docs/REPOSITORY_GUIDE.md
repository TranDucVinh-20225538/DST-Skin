# DST-Skin — Repository Guide

**Purpose of this document**: a full audit of the repository as it exists on disk (code, data, artifacts), for anyone picking up this research project. It is descriptive only — no source files were modified to produce it.

**Scope note on tracked vs. local state**: git history is a single commit (`1b0207f`, "Initial DST-Skin research codebase") containing only `src/`, `scripts/`, `README.md`, `requirements.txt`, `.gitignore`, and the orphaned root-level `split.py`. Everything under `data/`, `outputs/`, `experiments/`, `logs/`, `reader_images_96/`, and `*.pth`/`*.pt` checkpoints is excluded by `.gitignore` and exists only on this local machine. This guide covers both — the versioned code and the local data/output state — but the distinction matters if the repo is ever cloned fresh: **none of `data/`, `outputs/`, or `data/models/` ship with the repo.**

---

## 1. What this project is

A research pipeline for **out-of-distribution (OOD) detection / reliability triage in binary skin-lesion classification** (malignant vs. benign), branded in the codebase and outputs as **"DST-Skin"**. Three CNN backbones (ResNet-18, ResNet-50, EfficientNet-B3) are fine-tuned on ISIC 2018 (Task 3) as the in-distribution (ID) dataset, then a suite of post-hoc OOD/confidence scores — most centrally a **Mahalanobis distance in L2-normalized feature space** — is used to detect distribution shift when the same models are run on PAD-UFES-20 (treated as the OOD dataset). The end goal, evidenced by the reader-study and Grad-CAM scripts, is a clinical-triage narrative: use the reliability score to decide when to trust the AI's prediction vs. defer to a human reader, and to illustrate this with qualitative case studies (safe/trusted, dangerous trap, OOD-defer, OOD-rescue).

---

## 2. Repository architecture

```
DST-Skin/
├── README.md                  # canonical (but incomplete) pipeline instructions
├── requirements.txt           # unpinned deps (torch, sklearn, umap-learn, etc.)
├── split.py                   # ORPHANED — references a HAM10000 dataset not in this repo
├── src/                       # importable library code (PYTHONPATH=.)
│   ├── datasets/isic_dataset.py
│   ├── models/{resnet18,resnet50,efficientnet_b3}.py
│   ├── losses/                # EMPTY — scaffold, unused
│   └── utils/{feature_extractor,scoring,ood_vim_react,calibration}.py
├── scripts/                   # all executable entry points (23 files, flat, no subpackages)
├── configs/                   # EMPTY — scaffold, no config-driven runs exist
├── experiments/               # EMPTY — scaffold
├── logs/                      # EMPTY — scaffold, no logging framework wired up
├── data/                      # gitignored, local only
│   ├── raw/isic2018/…         # official ISIC 2018 Task 3 train+val images & ground truth
│   ├── raw/pad_ufes20/…       # PAD-UFES-20 images + metadata.csv
│   ├── processed/             # binarized label CSVs derived from raw
│   ├── models/                # 6 trained checkpoints (.pth)
│   └── reader_study/          # curated case-selection CSVs for the reader study
├── outputs/                   # gitignored, local only — all pipeline results
│   ├── features/              # 3 large .pt files: cached logits+features per backbone
│   ├── reports/                # CSV metrics + final figures (PDF/PNG)
│   ├── figures/                # triage & UMAP figures
│   ├── gradcam_examples/      # a small hand-picked Grad-CAM figure set
│   └── (root-level) results_*.csv, *_per_sample.csv   # legacy-pipeline outputs
└── reader_images_96/           # flat folder of 96 copied PNG/JPG images for the reader study
```

Design pattern: this is a **script-driven research repo**, not a package/framework. `src/` holds small, mostly stateless reusable building blocks (model factories, a generic dataset class, a feature-extraction hook utility, and an OOD-scoring class). `scripts/` holds one-off, largely copy-pasted-per-backbone drivers that import from `src/` and hardcode all paths and hyperparameters inline. There is no CLI argument parsing, no config system, and no orchestration (no Makefile/Snakemake/DVC) — the README's numbered command list *is* the pipeline definition.

---

## 3. Datasets

### 3.1 ISIC 2018 (Task 3) — in-distribution (ID)
- Location: `data/raw/isic2018/ISIC2018_Task3_{Training,Validation}_{Input,GroundTruth}/`
- Raw ground truth is 7-class one-hot (`MEL, NV, BCC, AKIEC, BKL, DF, VASC`).
- Binarized by `scripts/create_isic_split.py` (current) — malignant = `MEL + BCC + AKIEC > 0`, everything else benign — into:
  - `data/processed/isic2018_binary/isic_2018_binary_train.csv` (10,015 images)
  - `data/processed/isic2018_binary/isic_2018_binary_val.csv` (193 images)
  - There's also an un-suffixed `isic_2018_binary.csv` (10,016 lines incl. header, i.e. identical row count to `_train.csv`) — an earlier/duplicate full-train export, likely superseded by the `_train`/`_val` pair.
  - **Train/val split is the *official* ISIC 2018 Task 3 challenge split** (separate directories), not a random split performed by this code.
  - A near-duplicate of this binarization logic also exists as a **stray, un-integrated file** `data/raw/isic2018/ISIC2018_Task3_Training_GroundTruth/x` (a Python script with the filename `x`, not `.py` — not referenced by any pipeline step, presumably a leftover terminal paste).

### 3.2 PAD-UFES-20 — out-of-distribution (OOD) / domain-shift set
- Location: `data/raw/pad_ufes20/images/` (2,298 images) + `metadata.csv` (not read directly by any current script; already pre-binarized).
- Binarized to `data/processed/pad_ufes20_binary/pad_ufes_binary.csv` (2,299 lines) via **another stray script**, `data/raw/pad_ufes20/x` — malignant = `{MEL, BCC, SCC, ACK}` from the `diagnostic` column. Like the ISIC `x` file, this is not invoked by any script in `scripts/`; it was apparently run once, by hand, to produce the processed CSV, and left behind.
- **Important framing assumption**: PAD-UFES-20 is treated wholesale as "OOD" throughout the codebase. In reality it differs from ISIC along *two* confounded axes — acquisition modality (clinical smartphone photos vs. dermoscopy) *and* possibly different disease-prevalence — so "OOD" here really means "domain-shifted," not anomalous/unseen-class data. This framing is consistent throughout but is never stated explicitly in code or README.

### 3.3 Reader-study case sets
- `data/reader_study/reader_cases_effb3.csv` (75 cases) — an earlier iteration, columns `case_id, image_path, dataset, label, prediction, mahalanobis, reader_group, is_ood, correct`. `reader_group` uses a taxonomy of `AI_confident_correct` / `OOD_high_confident_wrong` / etc.
- `data/reader_study/reader_cases_selected_96.csv` (96 cases) — the current/final iteration, columns `case_id, image_path, gt_label, id_ood_flag, ai_pred, ai_prob_malignant, ai_prob_pred, ai_correct, dstskin_maha, reliability_flag, behavior_group, danger_score`. Produced by `scripts/select_reader_cases.py` (see §6).
- These two CSVs use **different, non-overlapping taxonomies** (`reader_group` vs. `behavior_group`) and are consumed by different downstream scripts (`dst_for_reader.py` vs. `grad_cam.py`/`collect_reader_images.py`) — see §7 for how they relate.

### 3.4 HAM10000 (referenced but absent)
`split.py` (repo root) processes `data/raw/ham10000/HAM10000_metadata.csv` into `data/processed/ham10000/ham10000_test_clean.csv`, explicitly filtering out any image ID already used in the ISIC train split (dedup against `isic_2018_binary_train.csv`) — i.e., it looks like preparation for using HAM10000 as a **third, cleaned external test set** (since ISIC2018 and HAM10000 overlap heavily). **No `data/raw/ham10000/` directory exists anywhere in the repository.** This script cannot currently run and no other script references its output. It is a dead/orphaned artifact, but a meaningful one — see §9.

---

## 4. Models (`src/models/`)

All three are thin factory functions wrapping `torchvision.models`, ImageNet-pretrained, with the final classification layer replaced for 2-class output:

| File | Function | Backbone | Head replaced | Feature dim (pre-fc) |
|---|---|---|---|---|
| `resnet18.py` | `get_resnet18(num_classes=2, pretrained=True)` | `torchvision.models.resnet18` | `model.fc` → `Linear(in, 2)` | 512 |
| `resnet50.py` | `get_resnet50(num_classes=2, pretrained=True)` | `torchvision.models.resnet50` | `model.fc` → `Linear(in, 2)` | 2048 |
| `efficientnet_b3.py` | `get_efficientnet_b3(num_classes=2, pretrained=True)` | `torchvision.models.efficientnet_b3` | `model.classifier[1]` → `Linear(in, 2)` | 1536 |

Note on API inconsistency: `resnet18.py` uses the deprecated boolean `pretrained=True` argument; `resnet50.py` already uses the modern `weights=ResNet50_Weights.IMAGENET1K_V1` API; `efficientnet_b3.py` uses `weights=EfficientNet_B3_Weights.DEFAULT`. Purely cosmetic/deprecation-warning risk, not a functional bug, but a marker that the three model files were not written/updated at the same time.

---

## 5. Training pipeline (`scripts/train_{resnet18,resnet50,efficientnet_b3}_robust.py`)

Three near-identical scripts (each self-contained, ~160 lines, no shared training loop module). Common structure:

- **Loss / optimizer**: plain `CrossEntropyLoss`, `Adam(lr=1e-4, weight_decay=1e-4)`, `CosineAnnealingLR` over `T_max=10` epochs. Fixed at **10 epochs**, no early stopping, no LR search.
- **Augmentation** ("robust" in the filename refers to this heavy augmentation, not to any adversarial-robustness or robust-loss technique): `Resize → RandomResizedCrop → RandomHorizontalFlip (+RandomVerticalFlip for R18) → ColorJitter → GaussianBlur → RandomAffine → ToTensor → ImageNet Normalize`. Slightly different crop scales/ColorJitter strengths per backbone (hand-tuned, not shared).
- **Resolution**: 224×224 for ResNet-18/50, 300×300 for EfficientNet-B3 (matches its native input size).
- **Batch size**: 32 (R18, B3) / 64 (R50). `num_workers=0` throughout (no multiprocessing data loading — likely tuned for a single local macOS/MPS machine).
- **Device selection**: `mps` → `cuda` → `cpu` fallback, i.e. developed primarily on Apple-Silicon MPS.
- **Model selection**: best checkpoint saved on highest validation AUROC (`{name}_robust_best.pth`); a final `{name}_robust_epoch_last.pth` is also always saved. Both exist for all three backbones in `data/models/`.
- **No class-balancing** (no weighted loss, no oversampling) despite malignant being the minority class in both ISIC and PAD-UFES — worth checking prevalence before trusting raw accuracy numbers.
- **Reproducibility gap**: no `torch.manual_seed`/`set_seed` call anywhere in the three training scripts — training runs are not seeded, so re-running will not reproduce the exact checkpoints currently in `data/models/`.

There is no validation-time logging beyond stdout `print()` — no TensorBoard, no CSV logger, no wandb integration (consistent with `logs/` being empty).

---

## 6. Feature extraction (`src/utils/feature_extractor.py`, `scripts/extract_once_*.py`)

`extract_features_and_logits(model, loader, device, target_layer=None)` is the single reusable extraction utility:
- Registers a **forward hook** on `target_layer` (default: `model.avgpool` if present, else `model.features[-1]`) — for all three backbones here this resolves to the global-average-pool layer, so the extracted "feature" is always the **pre-classifier pooled embedding** (512 / 2048 / 1536-dim).
- Runs one no-grad pass over a loader, flattening and collecting features and raw logits per batch, returns them as concatenated numpy arrays.
- Handles both `(img, label)` and `(img, label, path)` batch shapes (the latter is what `ISICDataset` actually returns).

`scripts/extract_once_{resnet18,resnet50,efficientnet_b3}.py` (near-identical, copy-paste-per-backbone) each:
1. Load the corresponding `*_robust_best.pth` checkpoint.
2. Run extraction over **ISIC train** (full 10,015), **ISIC val** (193), and **PAD-UFES** (2,298, used as OOD).
3. Save one bundled `.pt` file to `outputs/features/{name}_isic_pad_features.pt` containing: `train/val/ood_{logits,feats,labels}` plus the final linear layer's `fc_weight`/`fc_bias` (needed downstream for ReAct/ViM, which operate on the *logit-producing linear layer*, not the backbone itself).

All three header comments literally say `# scripts/extract_once_efficientnet_b3.py` regardless of the actual filename — a copy-paste artifact, harmless but a hint the ResNet files were cloned from the EfficientNet one.

These `.pt` files are large (33–134 MB) and are the sole input to all downstream scoring/analysis scripts — **the README explicitly warns "feature extraction scripts overwrite existing outputs" and "run extraction immediately before analysis,"** i.e. there is no versioning of which model checkpoint a given feature file corresponds to beyond the filename.

---

## 7. OOD / reliability scoring — the core method (`src/utils/scoring.py`, `src/utils/ood_vim_react.py`)

This is the intellectual core of the repo. `OODScorer` (in `scoring.py`) is fit once on ID training features and then scores arbitrary (logits, features) pairs:

**Fit (`OODScorer.fit`)**:
- L2-normalizes training features.
- Fits a **Ledoit-Wolf shrinkage covariance** estimator (`sklearn.covariance.LedoitWolf`) on the normalized features → gives a mean `mu` and a `precision` matrix, used for Mahalanobis distance. Shrinkage covariance is a sensible choice given feature dims (512–2048) can exceed or approach sample counts in ablation runs (see below).
- Fits a cosine-metric `NearestNeighbors` index (`k=50`) for k-NN distance scoring.
- Optionally fits **ReAct** (`fit_react_on_fc`, in `ood_vim_react.py`): clips features at the `p`-th percentile (default 90th) of train-feature activations, then computes energy from `clipped_feats @ W.T + b`.
- Optionally fits **ViM** (Virtual-logit Matching, `fit_vim`): finds a new origin from the linear layer's null space, computes the residual (non-principal) subspace of the training feature covariance, and calibrates a scale `alpha` so the "virtual logit" norm is comparable to real logit magnitudes. This is a faithful re-implementation of the official ViM algorithm (Wang et al., CVPR 2022).

**Score (`OODScorer.get_all_scores`)** — returns a dict, all convention **higher = more in-distribution/reliable**:
- `msp` — max softmax probability.
- `energy` — negative log-sum-exp of logits (sign-flipped to the higher-is-ID convention).
- `logit_norm` (actually returned as raw L2 norm of logits, *not* flipped) and a separate scalar `cvid_logit_norm` (coefficient of variation of the logit norm across the batch — a single float per call, not a per-sample score; flagged in code by scripts that skip any method starting with `"cvid"`).
- `mahalanobis` — **negative** Mahalanobis distance in normalized-feature space (this is the score referred to as `dstskin_maha` everywhere downstream and is the paper's headline metric).
- `knn` — negative mean cosine distance to 50 nearest training neighbors.
- `react_energy`, `vim` — only present if those flags were enabled at `fit()` time.

`compute_ece` (also duplicated in `src/utils/calibration.py` as a free function) implements standard 15-bin Expected Calibration Error.

**Empirical results** (from `outputs/reports/*_score_comparison.csv`, ISIC-val-vs-PAD-UFES AUROC/FPR95, current checkpoints):

| Backbone | Method | AUROC | FPR95 |
|---|---|---|---|
| ResNet-18 | MSP | 0.690 | 0.869 |
| ResNet-18 | Energy | 0.647 | 0.924 |
| ResNet-18 | **Mahalanobis** | **0.889** | **0.589** |
| ResNet-18 | k-NN | 0.830 | 0.705 |
| ResNet-18 | ViM | 0.771 | 0.661 |
| ResNet-50 | MSP | 0.629 | 0.908 |
| ResNet-50 | **Mahalanobis** | **0.935** | **0.244** |
| ResNet-50 | k-NN | 0.845 | 0.456 |
| EfficientNet-B3 | MSP | 0.760 | 0.814 |
| EfficientNet-B3 | **Mahalanobis** | **0.965** | **0.182** |
| EfficientNet-B3 | k-NN | 0.920 | 0.310 |

Mahalanobis distance dominates logit-based methods (MSP/Energy/ViM/ReAct) across all three backbones by a wide margin, and EfficientNet-B3 gives the strongest separation — consistent with the paper apparently centering on Mahalanobis ("DST-Skin" = **D**istance-**S**pace **T**riage for Skin, inferred from usage, not stated anywhere as an acronym) as the deployed score.

**Statistical caveat** (not handled anywhere in the code): the ID-side of this AUROC/FPR95 computation is only 193 samples (ISIC val) against 2,298 OOD samples — no bootstrap confidence intervals or significance tests are computed for any of these numbers anywhere in the pipeline.

---

## 8. Experiment / benchmark scripts — two generations coexisting

This is the most important structural finding for anyone continuing this work: **there are two parallel, non-identical benchmark pipelines**, and the second (current) one silently depends on an artifact only produced by the first (legacy) one.

### 8.1 Legacy pipeline — `scripts/run_journal_benchmark_{resnet18,resnet50,efficientnet_b3}.py`
- Self-contained: does its own on-the-fly feature extraction (doesn't read `outputs/features/*.pt`), fits a plain `OODScorer` (no ReAct/ViM), and writes:
  - `outputs/results_{backbone}_robust.csv` (summary AUROC/FPR95 per method)
  - **only the EfficientNet-B3 variant** also writes `outputs/efficientnet_b3_per_sample.csv` — a per-image CSV with `image_path, label, prediction, dataset, mahalanobis`.
- **Broken as-is**: `run_journal_benchmark_resnet18.py` hardcodes `data/models/resnet18_robust_epoch10.pth`, which does not exist in `data/models/` (only `_best.pth`/`_epoch_last.pth` are present) — this script will crash with `FileNotFoundError` if run today.
- `run_journal_benchmark_resnet50.py` correctly targets `resnet50_robust_epoch_last.pth` (exists); `run_journal_benchmark_efficientnet_b3.py` correctly targets `efficientnet_b3_robust_best.pth` (exists).

### 8.2 Current pipeline — `scripts/analyze_benchmark_{resnet18,resnet50,efficientnet_b3}.py`
- This is the one documented in the README. Reads the pre-extracted `outputs/features/*.pt` files (from §6), fits `OODScorer` with **both ReAct and ViM enabled**, and produces the richer, current set of reports:
  - `outputs/reports/{backbone}_score_comparison.csv` (table in §7)
  - `outputs/reports/{backbone}_mahalanobis_ablation_N.csv` — Mahalanobis AUROC/FPR95 as a function of how many ID training samples (`N ∈ {100, 500, 1000, 2500, 5000}`) the scorer is fit on (low-data-regime ablation)
  - `outputs/reports/{backbone}_risk_coverage_mahalanobis.csv` + `_summary.csv` — full risk-coverage curve on ISIC val, plus coverage achievable at target risk thresholds {10%, 15%, 20%}
  - `resnet18_score.py` and `resnet50_score.py` variants are byte-identical to each other; the EfficientNet-B3 variant differs only in filenames/print strings (confirmed via diff).
- Stray file: `outputs/reports/resne50_risk_coverage_summary.csv` (typo "resne50") sits alongside the correctly-named `resnet50_risk_coverage_summary.csv` — a leftover from a renamed/rerun script, safe to ignore but a minor housekeeping item.

### 8.3 The cross-dependency
`scripts/select_reader_cases.py` (current reader-study case selector, see §9) reads `CANDIDATE_CSV = "outputs/efficientnet_b3_per_sample.csv"` — **which is only produced by the legacy `run_journal_benchmark_efficientnet_b3.py`, not by anything in the current `analyze_benchmark_*` / `extract_once_*` pipeline.** So the "current" reader-study pipeline is not actually runnable end-to-end from the README's documented steps alone; the legacy EfficientNet-B3 benchmark script must also be run first to regenerate `outputs/efficientnet_b3_per_sample.csv`. This dependency is undocumented anywhere.

### 8.4 Orphaned report artifacts
`outputs/reports/final_predictions_for_triage.csv` and `outputs/reports/full_triage_data.csv` (both keyed on PAD-UFES `image_id`, with `prediction, prob_malignant, softmax_confidence[,mahalanobis_score]`) have **no corresponding generator script anywhere in `scripts/`**. They were almost certainly produced by an ad hoc/notebook script that was never committed. Treat these two files as non-reproducible from the current codebase.

---

## 9. Reader-study / clinical-triage pipeline

Two generations exist here too, mirroring §8:

1. **Earlier** (`data/reader_study/reader_cases_effb3.csv`, 75 cases, `reader_group` taxonomy) → `scripts/dst_for_reader.py` re-scores these cases with a freshly-fit `OODScorer` (Mahalanobis + k-NN only) against `outputs/features/effb3_isic_pad_features.pt`, adds a `reader_reliability_flag` (High/Low, with a special median-split rule for the `OOD_high_confident_wrong` group), and writes `outputs/reports/reader_study_effb3_with_ai.csv`. Nothing downstream consumes this file.

2. **Current** (`scripts/select_reader_cases.py`, internally headed `# scripts/select_reader_cases_v2.py`): builds a **96-case balanced set** across 4 target categories (24 each): `SAFE_TRUSTED`, `DANGEROUS_TRAP`, `OOD_SHOULD_DEFER`, `OOD_TRAP_RESCUE` (`ID_UNCERTAIN` is defined in the taxonomy but targeted at 0 cases). Group assignment logic (`build_behavior_groups`) is priority-ordered (OOD_TRAP_RESCUE checked first, SAFE_TRUSTED last) and a `danger_score = 0.7·ai_prob_pred + 0.3·norm(maha) − 0.5·ai_correct` is used to rank/backfill toward exactly 96 cases (`fill_to_target`). Writes:
   - `outputs/reports/reader_candidate_pool_with_ai.csv` (full 2,492-row scored candidate pool, ISIC val + all of PAD-UFES)
   - `data/reader_study/reader_cases_selected_96.csv` (final 96, this is the one actually used downstream)

Downstream of the 96-case set:
- `scripts/collect_reader_images.py` copies the 96 referenced images (by `image_path`) into a flat `reader_images_96/` folder (confirmed: exactly 96 files present) — presumably the literal image set shown to human reader-study participants, stripped of any AI metadata/filenames that would reveal ground truth.
- `scripts/grad_cam.py` picks 2 cases per behavior-group (8 total) from the 96-case CSV, generates Grad-CAM heatmaps (manual hook-based implementation, target layer = `model.features[-1]`, i.e. the last EfficientNet-B3 conv block) for each, and assembles a 4-row × 4-column qualitative figure at `outputs/reports/figure_gradcam_reader_study_main.{png,pdf}` (present in outputs). There is also a separate `outputs/gradcam_examples/` directory with 10 individually-saved Grad-CAM PNGs (`isic_confident_*`, `pad_ood_*`) plus `figure5_gradcam_final.pdf` — not generated by `grad_cam.py` as it currently exists (its naming convention and single-image-per-file layout doesn't match `grad_cam.py`'s output), so likely from an earlier/ad hoc invocation or a variant script not present in the repo. Treat as another non-reproducible artifact.

---

## 10. Visualization / figure scripts

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `scripts/plot_ablation.py` | `outputs/reports/{backbone}_mahalanobis_ablation_N.csv` (×3) | `outputs/reports/figure3_ablation_final.{pdf,png}` | 2-panel AUROC/FPR95-vs-N plot across all 3 backbones. Self-contained, runs cleanly given the CSVs exist. |
| `scripts/plot_triage_final_all.py` | Re-extracts features live from `data/models/*.pth` + raw images (does **not** read cached `outputs/features/*.pt`) | `outputs/figures/SOTA_TRIAGE_COMPARISON_FINAL.{png,pdf}` | Risk-coverage curve (risk vs. coverage, 100%→10%) on PAD-UFES, ranked by Mahalanobis reliability, all 3 backbones overlaid with a 15% clinical-risk-threshold reference line. **Broken as-is**: hardcodes `data/models/resnet18_robust_epoch10.pth` (same missing-checkpoint issue as §8.1). Internal header comment says `# scripts/run_risk_coverage_backbones.py` — apparent rename not reflected in the header. |
| `scripts/plot_umap_visual.py` | `data/models/efficientnet_b3_robust_best.pth` + live feature extraction | `outputs/figures/UMAP_VISUALIZATION_FINAL.png` | UMAP (`n_neighbors=50, min_dist=0.3, cosine`) of 1,000 ISIC-val + all 2,298 PAD-UFES embeddings, EfficientNet-B3 only. Self-contained, runs cleanly. |
| `scripts/grad_cam.py` | see §9 | `outputs/reports/figure_gradcam_reader_study_main.{png,pdf}` | See §9. |

**README gap**: the README's "Generate Figures → Triage" section instructs running `scripts/plot_triage_final_all.py` *and* `scripts/plot_triage_ood_final.py` — **the latter file does not exist anywhere in the repository.** Either it was renamed to `plot_triage_final_all.py` without updating the README, or it's an unfinished/uncommitted second figure script.

`outputs/figures/framework.png` is a standalone image with no generator script — almost certainly a manually-drawn architecture/pipeline diagram for the paper, not a pipeline output. `outputs/figures/final_paper/` and `outputs/figures/visualization/` are empty directories (contain only `.DS_Store`).

---

## 11. Saved artifacts inventory (current local state)

**Checkpoints** (`data/models/`, all gitignored): `{resnet18,resnet50,efficientnet_b3}_robust_{best,epoch_last}.pth` — 6 files, 41–90 MB each, standard `state_dict()` saves loadable via each model's factory function.

**Cached features** (`outputs/features/`): `{resnet18,resnet50,effb3}_isic_pad_features.pt` — 33–134 MB each, dict of numpy arrays + fc weights as described in §6.

**Reports** (`outputs/reports/`): per-backbone score-comparison, ablation, and risk-coverage CSVs (§8.2); final figures; reader-study CSVs (§9); two orphaned triage CSVs (§8.4); one typo'd stray file.

**Figures** (`outputs/figures/`, `outputs/gradcam_examples/`): triage comparison, UMAP, Grad-CAM figures as above, plus a manually-authored framework diagram.

**Reader-study assets**: `data/reader_study/*.csv` (case metadata) + `reader_images_96/` (flat image copies, gitignored).

---

## 12. Assumptions baked into the pipeline

- **Binary relabeling is a hand-picked clinical simplification**: ISIC's 7 classes → malignant iff `{MEL, BCC, AKIEC}`; PAD-UFES's diagnostic codes → malignant iff `{MEL, BCC, SCC, ACK}`. Note the two label sets aren't identical (AKIEC vs. ACK naming aside, PAD-UFES additionally includes SCC as malignant while ISIC's 7 classes don't have a distinct SCC category) — a deliberate but unstated harmonization choice between the two datasets' diagnostic taxonomies.
- **ISIC's official Task 3 val split (193 images) doubles as both the model-selection validation set *and* the OOD-benchmark's "ID" reference set** — the same 193 images are used to pick the best training checkpoint and then to compute every downstream AUROC/FPR95/risk-coverage number against PAD-UFES. This is a single, fixed, quite small ID evaluation set reused across every experiment in the repo.
- **PAD-UFES-20 = "OOD"** is treated as ground truth throughout, conflating domain shift (imaging modality) with genuine distributional/semantic novelty.
- **The last global-average-pooled layer is *the* feature representation** for every OOD method (Mahalanobis, k-NN, ReAct, ViM) — no other layer or multi-layer ensemble is explored.
- **Development environment**: Apple Silicon (`mps` device preferred), `num_workers=0` everywhere, no distributed/multi-GPU code path.
- Every script assumes it is invoked as `PYTHONPATH=. python scripts/<name>.py` from the repo root (per README) — no path is ever resolved relative to `__file__`.

## 13. Limitations / inconsistencies to be aware of

1. **Two coexisting benchmark pipelines** (`run_journal_benchmark_*` vs. `analyze_benchmark_*`) with a hidden cross-dependency (§8.3) — following the README alone does not reproduce the reader-study outputs.
2. **Two broken scripts** reference a checkpoint filename (`resnet18_robust_epoch10.pth`) that isn't present in `data/models/`: `run_journal_benchmark_resnet18.py`, `plot_triage_final_all.py`.
3. **README references a nonexistent script**: `scripts/plot_triage_ood_final.py`.
4. **Orphaned/non-reproducible artifacts**: `outputs/reports/final_predictions_for_triage.csv`, `outputs/reports/full_triage_data.csv`, and the 10-file `outputs/gradcam_examples/` set have no generator script in the repo.
5. **`split.py` (HAM10000 prep) and both `x`-named label-conversion scripts are dead code** — none are invoked by anything else, and `split.py`'s target dataset (`data/raw/ham10000/`) doesn't exist locally.
6. **No seeding in training scripts** — checkpoints in `data/models/` are not exactly reproducible by re-running `train_*_robust.py`.
7. **No class-balance handling** in training (plain cross-entropy) despite malignant being a minority class.
8. **Small, reused ID evaluation set** (n=193) with no confidence intervals reported anywhere for AUROC/FPR95 comparisons.
9. **Empty scaffold directories** (`configs/`, `experiments/`, `logs/`, `src/losses/`) indicate planned-but-never-built infrastructure — no config system, no experiment tracking, no logging framework, no custom loss functions, despite the directory structure implying they were intended.
10. **No tests, linting, or CI** anywhere in the repo.
11. **Unpinned dependencies** (`requirements.txt` uses `>=` everywhere) with no lockfile — exact package versions used to produce current checkpoints/results are not recorded.
12. **Mixed-language comments/prints** (Vietnamese + English) throughout `scripts/` — not a bug, but relevant if the repo is to be shared externally or with collaborators who don't read Vietnamese.
13. Minor cosmetic issues: copy-pasted file-path header comments in all `extract_once_*.py` files; typo'd `resne50_risk_coverage_summary.csv`; inconsistent `pretrained=` vs `weights=` API usage across model files.

## 14. Reusable components for future work

These are the parts of the codebase that are genuinely decoupled from the skin-lesion specifics and could be lifted into a new project or extended with minimal friction:

- **`src/utils/scoring.py::OODScorer`** — a clean, dependency-light (numpy/sklearn/torch) implementation of MSP, Energy, Mahalanobis (Ledoit-Wolf), k-NN, ReAct, and ViM scoring, all normalized to a "higher = more in-distribution" convention. Fully generic over any (features, logits) pair — not skin-imagery-specific at all.
- **`src/utils/ood_vim_react.py`** — standalone, correctly-implemented ReAct and ViM (residual-subspace) scorers; usable independently of `OODScorer`.
- **`src/utils/feature_extractor.py::extract_features_and_logits`** — a generic forward-hook-based extractor that works with any `torchvision`-style CNN with either an `avgpool` or `features` attribute; trivially reusable for other backbones/datasets.
- **`src/datasets/isic_dataset.py::ISICDataset`** — despite the name, this is a generic "CSV + image-folder" binary classification dataset (only requires `image` and `binary_label` columns); could be renamed/generalized for reuse beyond ISIC.
- **`src/models/*.py` factory pattern** — trivial to extend to additional torchvision backbones by following the same `get_<name>(num_classes, pretrained)` convention.
- **`scripts/grad_cam.py`'s hook-based Grad-CAM core** (`generate_gradcam`, `overlay_heatmap_on_image`) — self-contained, no dependency on the reader-study-specific case-selection logic; portable to any CNN with a nameable target conv layer.
- **The risk-coverage / FPR95 / AUROC helper functions** duplicated (identically, for R18/R50) across `analyze_benchmark_*.py` (`calc_auroc`, `calc_fpr95`, `risk_coverage`, `extract_coverage_at_target_risk`) are strong candidates for promotion into `src/utils/` as a shared module — currently triplicated with only filename/label changes.
- **The `danger_score` / behavior-group triage taxonomy** in `select_reader_cases.py` (`build_behavior_groups`, `fill_to_target`) encodes a reusable general pattern — "stratified case sampling by (correctness × confidence × reliability)" — that could be factored out for any human-reader-study design beyond this specific project.
- **`split.py` + the HAM10000-dedup logic** (excluding any ISIC-train image ID from a candidate external test set) is a sound pattern for adding further external test sets later, once/if HAM10000 data is actually acquired.

---

## 15. If you need to re-run the pipeline today

Following the README as-is will get you through feature extraction, current-pipeline benchmarking, and the ablation/UMAP figures. To reach the reader-study and full triage figures, based on the actual code dependencies traced above, the order is:

1. `create_isic_split.py` (only if `data/processed/isic2018_binary/*` needs regenerating — it already exists).
2. `train_{resnet18,resnet50,efficientnet_b3}_robust.py` (only if checkpoints need regenerating — they already exist; not seeded, so results will differ slightly from current ones).
3. `extract_once_{resnet18,resnet50,efficientnet_b3}.py` → refreshes `outputs/features/*.pt`.
4. `analyze_benchmark_{resnet18,resnet50,efficientnet_b3}.py` → current score-comparison/ablation/risk-coverage CSVs.
5. `plot_ablation.py`, `plot_umap_visual.py` → figures, no further dependencies.
6. For triage/reader-study figures: **first fix or re-run `run_journal_benchmark_efficientnet_b3.py`** (checkpoint path is valid) to regenerate `outputs/efficientnet_b3_per_sample.csv`, **then** `select_reader_cases.py` → `data/reader_study/reader_cases_selected_96.csv`, then `grad_cam.py` and/or `collect_reader_images.py`.
7. `plot_triage_final_all.py` currently needs its hardcoded ResNet-18 checkpoint path fixed (point it at `resnet18_robust_best.pth` or `_epoch_last.pth`) before it will run.
