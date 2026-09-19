# Precommit: Camelyon-8 × skin-8 — locked 2026-09-18

Apply official Kendall's W to **this file**. Do not reinterpret at the keyboard.
Code copy: `scripts/rebuild_architecture_invariance.py` (`JUMP_DELTA=0.15`, `FEATURE_DRAG=0.10`, `_scenario_camelyon`).
Do not overwrite `outputs/reports/NARRATIVE_LOCKS.txt` when rebuilding tables.

Related-work lock (do not widen): `outputs/reports/OPENMIBOOD_DIFFERENTIATION.md`.

---

## Before computing official Camelyon W

1. Zoo 8/8 CSVs on Camelyon (R18, R50, EffB3, DenseNet, ConvNeXt, MobileNet, RegNet, EffV2).
2. If EffV2 is still native **384**: run `@224` ablation first, same protocol as EffB3@224. **Do not mix native-384 into official W.**
3. Skin W=0.690 is **not** for the paper until EffV2-S@224 skin ablation (same resolution trap). Maha rank-1 on that 384 cell does not authorize quoting W.

Skin-8 is **not** scenario A. Maha rank-1 on 8/8 is cửa-2 evidence (stability across backbones). MSP-jump does **not** replicate on skin (0.68–0.82). Do not write “two-domain A”.

---

## A / B / C — Camelyon only, after n=8 + EffV2@224

ResNet mean = mean(MSP AUROC of resnet18, resnet50).
Jump if MSP − ResNet mean ≥ **0.15**.
EfficientNet family = `{effb3, efficientnet_v2_s}`.
“Other” = all remaining backbones.
Feature drag: Maha or kNN on a **new** backbone < median(R18, R50, EffB3) − **0.10**.

| code | rule | what it is |
|---|---|---|
| **C** | any new backbone drags Maha or kNN by >0.10 | checked **first** |
| **A** | not C, and MSP jump in ≥1 non-EfficientNet family | cross-family MSP jump |
| **B** | not C, jump only in EfficientNet family | architecture-specific anomaly |
| `no_jump` | not C, no jump ≥0.15 | EffB3 cell may be isolated |
| `pending_n8` | Camelyon backbones < 8 | wait |

Read the code label. Do not narrate a different letter.

---

## Floor revoked — Maha>MSP is not a paper

OpenMIBOOD (CVPR 2025) §5 already: feature-space post-hoc consistently outperforms logit/probability methods on medical imaging. Public before this manuscript. CMPB desk-reject “no novelty vs SOTA” now has a named source.

**Forbidden at every remaining tier** (workshop, CMPB-resubmit, CVPR main, NeurIPS E&D): a paper whose load-bearing claim is Maha>MSP / feature>logit.

Skin Maha 8/8 rank-1 and FM Maha ~0.995–1.0 may appear **under cửa 2** (replication + rank-stable across backbones / training paradigm). They are **not** a backup thesis.

**There is no floor that skips cửa 2 + Phao B.**

Old frame “cứt = negative/benchmark paper on Maha>MSP” is **revoked**.

---

## What each letter does (do not invent a fourth path)

**A (expected if unofficial 6/8 holds):**
- Main results = Camelyon-8 official W + skin-8 as Maha rank-stability (not MSP-jump) + Phao B (domain-dependent AUROC↔coverage, not “AUROC-winner = utility-loser”).
- Write the full manuscript. CVPR 2027 deadline **2026-11-16**.
- #2 (target-sample backbone protocol) and #3 (ensemble) in parallel, cheap.
- #4 (meta-feature → risk) when n=8.
- #1 (train recipe) **optional** on A; decide in 1–2 days against the two-month writing budget. Default **no** unless we explicitly want a method-paper.
- MIDOG zoo optional, not required to submit.

**B (EfficientNet-only jump):**
- Cửa 2 narrows: jump in a specific family, not a broad law. Phao B unchanged (independent of A/B/C).
- Do not push CVPR main as a general architecture-instability law. Workshop / CMPB-resubmit of **cửa 2-narrow + Phao B**, still **not** Maha>MSP.

**C (Maha/kNN dragged):**
- **No retreat to Maha>MSP.** That claim is spent.
- Activate Direction **#1** (architecture-agnostic train recipe, Camelyon pilot) and/or **#3** (ensemble) as rescue: detection **and** fix.
- If rescue fails: there is no remaining independent paper. Do not queue a Maha>MSP writeup. CMPB-resubmit is only viable if Phao B (2-domain protocol / coverage) still stands as ballast — not because Maha>MSP is novel.

**`no_jump`:** not pre-written. Stop and re-read this file; do not auto-promote to B or to Maha>MSP.

---

## Already closed (do not reopen)

- Phao A: GigaPath-Camelyon MSP 0.664, Phikon-Camelyon MSP 0.605, both >0.5. MIDOG-only footnote. File closed.
- Cửa 1 geometry; H4 n_train; CIFAR-10-C as a pillar; mixing OpenMIBOOD R50 ~0.59 with in-house R50 MSP 0.512; pooled Kendall W; reader study; extra MIDOG “why” hyps; skin on pathology FMs.
- EffB3 Camelyon MSP jump is architecture, not 300px (0.787 @224).

Phao B remains: rank instability + domain-dependent AUROC vs coverage@risk10% sign flip (Camelyon vs MIDOG). Combined thesis: `outputs/reports/combined_rank_utility_thesis.md`.

---

## Methods disclosure (body, not footnote)

First time MIDOG appears in methods, use this sentence:

> We build on OpenMIBOOD's public MIDOG split and ResNet-50 checkpoint as an external reference.

Public R50 ~0.59 = theirs. In-house R50 MSP 0.512 = ours. Never mix.

Toolkit (last, before submit): credit OpenMIBOOD for MIDOG imglists, public R50, and 1a / cs-ID / near-OOD protocol. Package only architecture zoo + FID-severity + OOD-triage coverage@risk. Do not re-sell their 24-method R50 table.

Locked related-work paragraph — copy from `OPENMIBOOD_DIFFERENTIATION.md`, do not paraphrase larger:

> OpenMIBOOD (CVPR 2025) ranks 24 post-hoc detectors on three medical benchmarks with a **single frozen classifier per domain** (ResNet-50 on MIDOG, ResNet-18 on PhaKIR, R(2+1)D on OASIS-3). It shows that ImageNet method ranking does not transfer to medical data and that feature-space scores outperform logit scores under that fixed-architecture protocol. It does not measure whether the detector ranking itself is stable across architectures. We keep their MIDOG split and public ResNet-50 checkpoint as an external reference, and ask the orthogonal question: **holding the dataset fixed, how much does detector ranking move when the backbone changes?**

---

## Venues (calendar)

| | |
|---|---|
| CVPR 2027 paper | **2026-11-16 AoE** — current target |
| CVPR 2027 registration | 2026-11-10 AoE |
| NeurIPS **2027** Evaluations & Datasets | pinned E&D backup (CFP not out; historically ~May 2027) |
| NeurIPS 2026 E&D | **closed 2026-05-06**; notify 2026-09-24 — do not plan against it |
| ICLR 2027 | abstract 2026-09-18 / paper 2026-09-25 — too late |
| ICML 2027 | CFP not out; historically ~Jan 2027 — first post-CVPR ML venue |

#2 (protocol) can start now; it does not wait for 8/8. Phao A has nothing left to wait for.

## Direction #2 — ID-only binary gate (locked 2026-09-18 02:17)

Revoked: T-scaling-monotonicity as motivation; convex MSP+Maha mix as the deliverable.

Keep: scalar-monotone f(MSP) cannot change AUROC (CDF/rank/log). Nothing else.

Gate, not mix:
1. Bar = Maha and ViM, not MSP. Selecting Maha on almost all Camelyon backbones = “ID-only selector reconfirms Maha.” Do not name a new detector.
2. Report AUROC spread + coverage@risk. Do not report Kendall W of the gate.
3. Fit on ID-calib (50% of ID-val, seed=42). Eval ID-val is the other 50% vs OOD. No OOD labels at fit. Precommit: Spearman(MSP, Maha) on calib; ρ < 0.5 → Maha else MSP.
4. CPU on saved features. Do not block Job B.

## Direction #2 — ID-only binary gate — CLOSED 2026-09-18 02:29

Job 59689. Precommit held. **Not a method.**

Camelyon chose Maha 2/8 (R18 ρ=0.499, R50 ρ=0.408); skin 0/8.
Mean AUROC Camelyon: gate 0.837, Maha 0.940, ViM 0.791. Spread gate 0.080 > Maha 0.040.
Do not retune ρ. Do not convex-mix. Only remaining method candidate is Job B.

Skin EffV2-S@224 (59686): MSP 0.737 / Maha 0.890 / kNN 0.849. No MSP-jump vs 384. kNN rank-7 at 384 is a resolution artifact. **Official W still waits Camelyon EffV2-S@224 (59687).**

## Official W + freeze — CLOSED 2026-09-18 10:50

Rebuild used EffV2-S@224 on Camelyon and skin. Native-384 dropped. ViT / SupCon / iWildCam are not W columns.

| domain | n | Kendall W |
|---|---|---|
| Camelyon17 | 8 | **0.694** |
| Skin ISIC→PAD | 8 | **0.791** |
| MIDOG (in-house) | 3 | 0.857 |

**scenario=A.** MSP jump 6/8 (all non-ResNet), including EffV2-S@224 (MSP 0.778, +0.233 vs ResNet mean 0.545). Maha or kNN rank 1–2 on all 8 Camelyon cells. Skin is Maha rank-1 on 8/8, **not** a second A.

Job B **DEAD** (59688): gap 0.308→0.238 (23%). R18 MSP 0.574→0.515 under SupCon — the shrink is partly R18 getting worse. No method candidate. No coverage-loss stretch.

**MIDOG zoo reopened 2026-09-18 (execution risk, not a new method).** Same script/protocol, 5 missing CNN backbones, in-house only. Do not mix OpenMIBOOD public R50. Official EffV2-S@224. Re-analyze original 3 with 7 methods so W is defined. This is the remaining “làm thêm” that can take MIDOG from n=3 to n=8 for cửa 2 and Phao B. Deadline 16/11 — ~8–9 weeks. Do not open a new loss.

Held-out, not W: ViT-B/16 HIT (MSP 0.884); iWildCam MISS (jump −0.029) = biomedical bound, one limitations sentence.

Write from `manuscript/cvpr2027_outline.md`. Do not reopen A/B/C.

---

## Method stretch — design risk, not execution risk (locked 2026-09-18 02:06)

Zoo / W / resolution / risk-coverage = **execution risk**. A new training objective = **design risk**: may not work, may not converge, may lose 2–3 weeks and still not beat CE or CE+SupCon.

**Do not open a new loss in parallel with writing or with Job B.**

Order:
1. Job A (T* vs jump) — **CLOSED 2026-09-18**. r(T*,jump)=0.268, p=0.52; Spearman=0; Dense−R18 MSP gap 0.308→0.308 after T*. Scale/calibration dismissed empirically. T-scaling is **not** a monotone of MSP (EffV2 0.726→0.744). No T-scale in recipe. No feature-norm/entropy hunt. Do not write a monotonicity theorem that forbids T-scaling AUROC change.
2. Job B = CE+SupCon recipe (Khosla 2020, λ=0.1 τ=0.07), R18 + DenseNet only. **Unblocked; running 59688.** Kill if DenseNet−R18 MSP gap does not shrink ≥50% (0.308 → ≤0.154). This is the method floor if it works: diagnostic + off-the-shelf mitigation.
3. **Only if B works**, optional stretch: a surrogate that targets **coverage@risk stability across backbones** (Phao B), not generic compactness (that is just SupCon again). Coverage@risk is discrete; needs a differentiable surrogate (soft-rank / pairwise ranking). Hard stop: if it does not beat the **Job B recipe** on the same R18/DenseNet gap (or on coverage@risk sign-stability) by a named date, drop it. Do not extend into writing time.

The paper does **not** require this stretch. Findings (cửa 2 + Phao B) ± Job B recipe is the submitable object. A new coverage loss must not become a condition for having a paper.

Not this stretch: center-loss, triplet, or “SupCon with a new name.” If B fails, write-only — do not invent a harder loss to rescue B. C still forces #1/#3 as rescue; that rescue is Job B / ensemble, not the coverage surrogate.

---

## #4 / #5 — execution-risk ceiling (locked 2026-09-18 02:46)

Canonical file: `decision_precommit_vit9_iwildcam.md`. Hard stop **2026-10-02**. Do not steal GPU from Job B / Camelyon EffV2@224; queue behind them.

**#5 iWildCam (heavier CVPR lever):** R18 + R50 + DenseNet-121, same jump rule, W not pooled. Hit → finding is not a medical benchmark. Miss → bound (“may be biomedical-specific”), not a failed paper. FMoW only if iWildCam data is blocked.

**#4 ViT-B/16 @224 Camelyon (not another CNN, not Swin-T).** Depth/param/RF regression on n=8 is **invalid** (LOO R² = −5.15). Do not use it. Precommit from the only n=8 rule that fits: non-`resnet_like` jumped 6/6 → **ViT-B/16 MSP ≥ 0.6947**. Miss = CNN-family bound. Do not refit after seeing the number.
