# Precommit: foundation-embedding group-leakage gate (foundation_gate)

Date: 2026-10-07 (written ~11:10 +07). Branch `rigor-pack`, code HEAD at writing 7fada85. Committed before
any foundation-model (FM) feature, probe, score or AUROC of this gate is computed. Not edited after results;
deviations go to a separate dated file `precommit_foundation_leakage_gate_2026-10-07.deviations.md` (so the
deposited sha256 stays valid).

Known before writing (already reported): all Camelyon17 rigor-pack / mechanism results (CNN Δ_fit, F2
works on 2/3 datasets), medbench results (DermaMNIST Δ_fit ViM 7/8, Maha 5/8 archs; ReAct Δ_fit ≈ 0), the
cached weights on the HPC (inventory below). No FM number for Camelyon17 or DermaMNIST under this protocol
has been computed. The earlier `fm_ood_pilot.py` FM run (UNI / Phikon / GigaPath standard AUROCs, no
group protocol) exists in `outputs/reports/fm/`; it is not used by any bar here.

Unchanged: every earlier precommit and bar; published scorer config
`OODScorer(k_nearest=50, use_react=True, react_percentile=90.0, use_vim=True, vim_dim=None)` (keys msp, energy,
logit_norm = ELogitNorm, react_energy, vim, mahalanobis = Ledoit-Wolf float64, knn = k 50 cosine);
`calc_auroc` (ID = higher score); float64 features in every scorer fit.

Claim lock (not expanded by this gate): group leakage in ID splits inflates feature-score OOD AUROC.
This gate only asks whether the leaky-ID inflation of feature scores and the flat ReAct pattern appear
under frozen FM embeddings at seed 42. Stain / Macenko / grayscale ablation is out of scope.

## L1. Datasets (priority order; cut only by STOP reasons)

| priority | dataset | role | groups / split |
|---|---|---|---|
| must | Camelyon17 (WILDS v1.0) | primary gate | slide; train (302,436) / id_val (33,560) / hospital-2 test = OOD (85,054), same index order as the rigor-pack indexed features; slide-disjoint 2-fold = H11c folds (`leakfree_fit_scores.folds`, seed 42), as `mech_cpu.load_cell` |
| then | DermaMNIST (MedMNIST+ 224) | secondary medical | medbench Phase-0 splits and arms unchanged (`medbench_common.arm`): M_std (std) and lesion-disjoint (B) arms b0 / b1; OOD = PAD-UFES-20 (primary), BCN-not-in-HAM and BloodMNIST secondary; images from `data/staged/dermamnist_256.tar` |
| optional | iWildCam (WILDS) | non-medical control | run only if Camelyon + DermaMNIST finish with projected total < 10 GPU-h; multibench folds |
| skip | RxRx1 | — | near-chance in multibench; not run |

STOP per dataset: missing data / download > 100 GB / licence click / DUA → skip, record, continue. No dataset
substituted after results. Nothing is registered or accepted on the user's behalf.

## L2. Backbones

CNN anchors (no retraining, no re-extraction): Camelyon ResNet50 and ConvNeXt-T seed 42 = the existing
mechanism cells (`outputs/reports/rigor_pack/mechanism_fix/cells/camelyon_{resnet50,convnext_tiny}.json`,
same H11c 2-fold protocol) and, for the logit within-model gap, `m4logit_camelyon_*.json` (ISBI-patch-2 v2
retrained seed-42 models). DermaMNIST anchors: medbench ResNet50 / ConvNeXt-T seed 42 scores (std, b0, b1).

Foundation models (frozen encoder, no fine-tune), attempted in this order; each either loads or gets a
STOP reason (gated / missing / OOM / import failure) and the grid continues:

1. DINOv2 ViT-B/14 (required if it loads): `torch.hub` `facebookresearch/dinov2` `dinov2_vitb14` (weights
   cached: `dinov2_vitb14_pretrain.pth`); embedding = normalised CLS token (`x_norm_clstoken`), 768-d.
2. UNI (ViT-L/16, `MahmoodLab/uni`, cached; `src/models/pathology_fm.py` spec "uni"); CLS, 1024-d. UNI2-h is
   not run (weights not cached; one UNI only, decided now).
3. CONCH: first `MahmoodLab/CONCH` (v1, gated; needs the `conch` package, installed into torch-env if pip /
   download work); if v1 cannot be loaded, CONCH v1.5 from the cached `MahmoodLab/TITAN` repository
   (`conch_v1_5.py` `build_conch`, its own eval transform); else STOP. Embedding = the image-encoder output
   used by the official repo for tile features.
4. Optional DINOv2 ViT-L/14 (`dinov2_vitl14`), only if the post-smoke projection allows (L6).

Transforms: each FM's published eval transform (DINOv2 / UNI: Resize 224 bicubic + CenterCrop 224 + ImageNet
mean / std; CONCH: the transform returned by its loader). The same eval transform for train, ID and OOD
(no augmentation). Weights float32, no autocast. Feature = penultimate embedding as above.

Classifier / logits (needed for MSP / Energy / ELogitNorm / ReAct / ViM): the existing
`fm_ood_pilot.fit_linear_head` recipe — `LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")` on the
raw (unstandardised) training embeddings, logits = X W^T + b (binary → [−z, z]). No other head recipe.
Seed: 42 only (lbfgs is deterministic; seed recorded). Fold / subsample RNG as medbench / mechanism.

Explicit STOP — multi-seed: no seed sweep in this job. After a PASS the only allowed action is writing
`PROMPT_SEED_SWEEP_FOUNDATION_2026-10-07.md`.

## L3. Protocol

Camelyon17, per loaded FM:
- Standard: scorer fit on all training embeddings, probe on all training embeddings; ID = id_val patches
  with a fold (as mech_cpu), OOD = hospital 2; 7 AUROCs.
- A-fit (leaky vs disjoint, identical to `mech_cpu.run_cell` 2-fold part / H11c): fit the scorer on fold-f
  training slides; same = AUROC on id_val patches of fold-f slides, disjoint = id_val patches of fold-(1−f)
  slides, OOD all; fold mean. Δ_fit = same − disjoint for Mahalanobis / kNN / ViM / ReAct (probe = full
  probe for the ViM / ReAct logit parts, as in mech_cpu).
- Logit within-model gap (the leaky-vs-disjoint contrast for MSP / Energy / ELogitNorm): the probe refitted on
  fold-f training embeddings; gap = AUROC(id_val fold f) − AUROC(id_val fold 1−f), OOD all, fold mean; ID
  accuracy of the fold probe on seen (fold f) and unseen (fold 1−f) id_val. Unseen-group accuracy < 0.8 →
  confound note.
- Uncertainty: cluster bootstrap B = 2000 (resample id_val slides, OOD images; `default_rng(2)`), 95%
  percentile CI of every Δ_fit and logit gap.
- Near-chance: a score whose AUROC is in [0.45, 0.55] under both same and disjoint is excluded from bar
  denominators (reported).
DermaMNIST, per loaded FM: frozen embeddings of every set of the medbench arms std / b0 / b1; probe per arm
on that arm's training images; `medbench_scores.py` unchanged (class-matched A-gap, A-fit Δ_fit, cluster
bootstrap B = 2000, B within-model gap), gates of medbench L5 G2 / G3 (G1 does not apply to a convex probe;
probe training accuracy reported). Reported next to the CNN anchors; not part of the gate.
Threads: extraction jobs OMP / OPENBLAS / MKL = 8; CPU scoring jobs = 16 (as mechanism); torch threads =
the same value.

## L4. Gate (Camelyon17)

Feature family = Mahalanobis, kNN, ViM; ReAct separately. n = loaded FMs.
- (1) Inflation: an FM "shows inflation" if >= 2 of its 3 counted feature scores have Δ_fit > 0.02. (1)
  holds if >= half of the n FMs show inflation, OR the median (over FMs) of each FM's median feature Δ_fit
  has the same sign as the CNN anchors' median and is not noise: > 0.01 and, for >= half of the FMs, >= 2 of
  3 feature-score Δ_fit CIs exclude 0.
- (2) ReAct flat: median over FMs of |Δ_fit(ReAct)| <= 0.02.
- PASS = (1) and (2). FAIL = (1) or (2) fails after every loaded FM is scored. n = 0 → INCONCLUSIVE (no FM
  weights). Ranking (winner, Kendall tau-b of the 7 scores, same vs disjoint) reported, not required.

## L5. After the gate

PASS → write `PROMPT_SEED_SWEEP_FOUNDATION_2026-10-07.md` (3 seeds × passing FMs × Camelyon only; frozen
features reused; only probe / scorer RNG re-fitted; no new dataset; no stain; QOS <= 4); do not submit it.
FAIL / INCONCLUSIVE → README states it; stop; no seed expansion; no extra backbones after seeing Δ.

## L6. Budget

Estimate: Camelyon extract DINOv2-B + UNI + CONCH ≈ 3-6 GPU-h, optional DINOv2-L +1-2, DermaMNIST × FMs
≈ 0.5-2, optional iWildCam 1-3, scoring CPU. Re-projected after the smoke (one FM, Camelyon subset). If
> 16 GPU-h, cut only in this order: (1) iWildCam, (2) DINOv2-L, (3) CONCH if not started. Never cut
Camelyon, DINOv2-B or UNI if they load, or DermaMNIST.

## Order and outputs

REPRO (Camelyon ResNet18 / ResNet50 seed-42 `leakfree_knn.py` vs committed CSV to 1e-6; job submitted before
this file) → this precommit (+ DEPOSIT) → inventory / load attempts → smoke → Camelyon grid → DermaMNIST →
optional iWildCam → gate. Outputs `outputs/reports/rigor_pack/foundation_gate/`: inventory.md,
access_status.json, summary.md/.csv, tables.json, gate_decision.md, README.md (GPU-h estimate vs actual,
skips, commit hashes, seed-sweep pointer), checklist_note.md. No home paths, data, tars, feature arrays
or weights committed.
