# CVPR 2027 outline — for review before drafting sections

Target: CVPR 2027 paper **2026-11-16 AoE**. Backup: NeurIPS 2027 E&D (not 2026).
Numbers frozen 2026-09-18: `outputs/reports/OFFICIAL_W_FREEZE.txt`. Official W does not change. Allowed GPU exceptions: 4-anchor seed-variance; MIDOG zoo n=3→8; shift-type extract (`decision_precommit_shift_type.md`); matched-recipe (`decision_precommit_matched_recipe.md`); Viên 1 stain-covariate (`decision_precommit_stain_cov.md`). Those runs never enter Kendall W. CVPR 2027 vs NeurIPS is decided after Viên 1, not today.

Working title (replace if needed): **Detector ranking is not architecture-invariant under medical covariate shift**

Load-bearing claim: holding the dataset fixed, post-hoc OOD **method ranking moves with the backbone**. Not “we beat MSP with Maha.” OpenMIBOOD already owns feature>logit on medical with a frozen classifier.

---

## 0. What this paper is / is not

**Is:** a characterization of architecture-sensitive post-hoc ranking **under medical covariate shift** (not a discovery that ranking can move with architecture). Camelyon-17 hospital shift as the founding cell (scenario **A**): MSP AUROC jumps ≥0.15 vs ResNet mean on 6/8 CNN families; Mahalanobis and kNN stay rank 1–2. Skin ISIC→PAD as Maha rank-stability **without** MSP-jump. Phao B: AUROC vs coverage@risk10% disagreement whose **sign flips** across Camelyon vs MIDOG. ViT-B/16 as a **pre-registered** hit outside the CNN family. iWildCam as a negative bound. A robustness appendix of failed alternative explanations.

**Is not:** the first observation that detector ranking depends on architecture; a new detector; a Maha>MSP paper; a two-domain A; a natural-image / semantic-OOD law; a method-paper (SupCon and ID-gate are dead); an OpenMIBOOD 24-method rerun.

Calibration: **~40–50% main-track.** Not because the numbers moved — they did not. Because the story is “characterize a known phenomenon in a specific setting (covariate / medical) with zoo-scale concordance, a structured MSP jump, and an operational AUROC↔coverage mismatch that the semantic-OOD cousins do not measure,” which is one notch weaker than “we found the phenomenon.” Workshop / CMPB still the high-confidence floor.

---

## 1. Introduction

**Job of the section:** name architecture as a hidden factor **in medical covariate-shift OOD evaluation**; locate that against OpenMIBOOD (frozen classifier) and against semantic-OOD cousins (ranking already moves with architecture on natural images). Do not open as a discovery of the phenomenon.

**Open with the Camelyon cell, not “nobody knew.”** Same hospital-2 split, same 7 scores: ResNet-18/50 MSP AUROC 0.57/0.52 vs DenseNet 0.88 / EffB3 0.80. A practitioner who copies an OpenMIBOOD ResNet-50 recipe on this shift gets a different detector ranking than one who uses DenseNet.

**Contributions (four bullets, no fifth):**

1. **Characterization, Camelyon-17, n=8, Kendall W = 0.694.** Under this covariate split, MSP jump is cross-family (6/6 non-ResNet; 4 families beyond EfficientNet). Feature-space scores stay top-2. This is not a claim that architecture-sensitive ranking is new.
2. **Domain asymmetry.** Skin ISIC→PAD (n=8, W = **0.791**): Maha rank-1 on 8/8, MSP stays in 0.68–0.74, **no** jump. Do not write “two-domain A.”
3. **Utility mismatch is unpredictable.** AUROC rank and coverage@risk10% rank disagree on Camelyon **and** MIDOG, but the direction flips. Action: do not pick a backbone for triage from an AUROC table.
4. **A locked prediction.** Family-prior precommit, before training: ViT-B/16 Camelyon MSP ≥ 0.6947. Observed **0.884**. Held out of the 8-CNN W on purpose.

**Do not put in the intro:** Maha>MSP as a finding; unofficial mixed-res W=0.666; native-384 EffV2; Job B; any wording that we were first to see ranking move with architecture (including teasers). iWildCam: at most one clause, or save for limitations.

**Do not claim:** first medical OOD benchmark; first feature>logit on medical; first architecture-sensitive OOD ranking.

---

## 2. Related work — lock this paragraph, then stop

Copy verbatim from `outputs/reports/OPENMIBOOD_DIFFERENTIATION.md`. Do not paraphrase larger:

> OpenMIBOOD (CVPR 2025) ranks 24 post-hoc detectors on three medical benchmarks with a **single frozen classifier per domain** (ResNet-50 on MIDOG, ResNet-18 on PhaKIR, R(2+1)D on OASIS-3). It shows that ImageNet method ranking does not transfer to medical data and that feature-space scores outperform logit scores under that fixed-architecture protocol. It does not measure whether the detector ranking itself is stable across architectures. We keep their MIDOG split and public ResNet-50 checkpoint as an external reference, and ask the orthogonal question: **holding the dataset fixed, how much does detector ranking move when the backbone changes?**

Then, short neighbors. **Do not write “nobody measured ranking vs backbone on natural images.”** A dedicated search (2026-09-18) found that they did; cite them and shrink the claim.

- **OpenOOD v1 (NeurIPS 2022 D&B):** architecture **is** fixed on purpose (LeNet / R18 / R50). That sentence is about v1, not the field.
- **OpenOOD v1.5 (Zhang et al.):** ImageNet-1K post-hoc on **ResNet-50, ViT-B/16, and Swin-T**. Fig. 3: “different post-processor may favor different architecture” (ASH/ReAct drop on transformers; RMDS prefers them). Three models, torchvision frozen, **semantic** near/far OOD, no Kendall W, no 8-CNN family zoo, not covariate hospital shift. Closest natural-image cousin. Cite it. Do not pretend v1.5 still fixes architecture.
- **Szyc, Walkowiak, Maciejewski (UAI 2023):** detector **rankings change** across three ResNet-101/110 CIFAR implementation variants with matched closed-set acc, and across seeds (feature methods especially). A reliability paper: SoTA ranks are not stable under uncontrolled experimental details. Not an 8-family zoo holding one shift fixed; CIFAR semantic OOD.
- **Datko, Szyc, Walkowiak, Maciejewski (ESWA 2026):** preferred post-hoc detector **per CNN/ViT line**, claimed stable across OOD benchmarks for that model. Architecture-specific winner, natural images. Opposite framing from a single-split concordance statistic.
- **Claros Olivares & Brockmeier (arXiv 2511.11934, 2025–26):** scratch CNN vs fine-tuned ViT; ranking cliques of confidence scores by shift severity (CIFAR / TinyImageNet). Two representation paradigms, many OOD regimes, method-selection guide. Not 8 CNN families on one covariate split.
- **OoD-Bench:** OOD *generalization*, not detection. Venue precedent only.
- **WILDS Camelyon17 / iWildCam:** covariate shift by hospital / camera trap. iWildCam is a bound, not a third main domain. That bound is what we have on natural-image **covariate** shift; do not cite OpenOOD Fig. 3 as if we had already tested ImageNet.

**What remains ours after this search:** not the existence of architecture-sensitive ranking. That is published on natural images. Ours is: **one medical covariate-shift split, held fixed; 8 CNN families + a precommitted ViT; Kendall W; a structured MSP jump (ResNet vs the other six) with feature methods rank 1–2; skin non-replication; iWildCam miss.** Reviewer weapon: “OpenOOD v1.5 Fig. 3 already.” Counter: three models, semantic ImageNet, qualitative; we measure concordance under hospital shift at zoo scale and show the pattern is not a general natural-image law (iWildCam).

**Forbidden related-work sentences:** list in `OPENMIBOOD_DIFFERENTIATION.md`, plus: “nobody has measured detector ranking vs backbone on natural images”; “OpenOOD v1.5 still fixes architecture.”

---

## 3. Method / setup

### 3.1 Shifts

| domain | ID | OOD | role |
|---|---|---|---|
| Camelyon17 WILDS | hospitals 0,3,4 | hospital 2 (`test`) | founding cell; official W n=8 |
| Skin ISIC→PAD | ISIC | PAD-UFES-20 | rank-stability without MSP-jump |
| MIDOG | OpenMIBOOD 1a | 1b+1c + remaining near-OOD | Phao B + external R50 reference; **n=3 only** |

First MIDOG mention in methods, not a footnote:

> We build on OpenMIBOOD's public MIDOG split and ResNet-50 checkpoint as an external reference.

Public R50 AUROC ~0.59 = **theirs**. In-house R50 MSP 0.512 = **ours**. Never mix.

### 3.2 Backbones (official W)

Eight CNN-family models, ImageNet init, 10-epoch fine-tune, seed 42, train-frac 1.0 on Camelyon. Recipes in `src/models/cnn_family.py` (not forced-identical Adam). **State this in methods; do not imply a locked-everything-but-architecture ablation.** Optimizer/schedule confound is limitations item 6. Only resolution was isolated (EffB3 @224/@300, EffV2-S @224/@384).

Resolution rule, one sentence: official EffV2-S is **224** on both Camelyon and skin (native 384 dropped). Official EffB3 Camelyon remains **300**; EffB3@224 is the appendix control that the jump is not a 300px artifact (MSP 0.787).

**Held out of W, reported separately:** ViT-B/16 @224 (precommit). Not Swin-T.

### 3.3 Scores and statistics

Seven post-hoc scores. Primary concordance: **Kendall's W per domain, never pooled**. Jump: MSP − mean(R18, R50) ≥ 0.15. Feature drag (scenario C, checked first): Maha or kNN on a new backbone < median(R18, R50, EffB3) − 0.10. Code: `scripts/rebuild_architecture_invariance.py`.

Coverage@risk10%: OOD-triage, script `pathology_risk_coverage.py`. ID-selective Camelyon is vacuous (ID-val acc ≈ 99.5%) — do not cite it.

### 3.4 What we tried and killed (pointer only here)

One sentence in methods: “We tested whether the MSP jump is an artifact of input resolution, temperature scaling, ID-only detector selection, or a CE+SupCon recipe; none of these accounts for it (Sec. 5 / appendix).” Details live in analysis + appendix, not here.

---

## 4. Results

### 4.1 Camelyon-17 — scenario A (main table)

Official Kendall W = **0.694** (n=8, 7 methods). ResNet mean MSP = 0.545. Jump iff MSP ≥ 0.695.

| Backbone | input | Maha (rank) | kNN (rank) | MSP | jump |
|---|---|---|---|---|---|
| ResNet-18 | 224 | 0.956 (1) | 0.936 (2) | 0.574 | no |
| ResNet-50 | 224 | 0.896 (1) | 0.866 (2) | 0.515 | no |
| DenseNet-121 | 224 | 0.988 (1) | 0.973 (2) | 0.883 | yes |
| ConvNeXt-Tiny | 224 | 0.928 (1) | 0.916 (2) | 0.846 | yes |
| MobileNetV3-L | 224 | 0.870 (2) | 0.904 (1) | 0.771 | yes |
| RegNetY-3.2GF | 224 | 0.964 (1) | 0.956 (2) | 0.784 | yes |
| EfficientNet-B3 | 300 | 0.885 (2) | 0.934 (1) | 0.803 | yes |
| EfficientNetV2-S | **224** | 0.880 (1) | 0.856 (2) | 0.778 | yes |

MSP jump occurs in **6/6 non-ResNet backbones across 4 distinct families beyond EfficientNet** (DenseNet, ConvNeXt, MobileNet, RegNet) → **A**, not B. (The A/B/C code threshold is ≥1 non-EfficientNet family; the evidence is stronger than that threshold. Write the 6/6 sentence in prose.) If seed-variance lands in the middle branch (mean jump ≥0.15 but 2+ seeds miss), add a variance clause here — do not rewrite A. Maha/kNN not dragged → not C. Feature methods occupy ranks 1–2 on every backbone (Maha #1 on 6/8; kNN #1 on EffB3 and MobileNet).

Logit-method spread (MSP 0.130) >> feature-method spread (Maha 0.045, kNN 0.041). Concordance is incomplete (W=0.694) **because** logit ranks move.

### 4.2 Skin ISIC→PAD — not a second A

Official W = **0.791**. Maha rank-1 and kNN rank-2 on **all 8**. MSP 0.68–0.74. EffV2-S@224 MSP 0.737 (native-384 was 0.823 / kNN rank 7 — resolution, not architecture). Write as **rank-stability of feature scores**, replication of OpenMIBOOD’s feature>logit under a zoo, not as MSP-jump.

### 4.3 Robustness check — resolution is not the Camelyon jump

One dedicated paragraph, not folded into 4.1:

- EffB3 Camelyon: 0.803 @300 → 0.787 @224. Jump survives.
- EffV2-S Camelyon: 0.726 @384 → **0.778 @224**. Jump survives and is not a 384 artifact.
- Skin EffV2-S: the @384 kNN collapse **is** a resolution artifact and goes away at 224.

Camelyon A is robust to the two backbones that invited a resolution objection. Skin’s EffV2 anomaly was the actual artifact. Do not merge these into one “resolution doesn’t matter” slogan.

### 4.4 Held-out prediction — ViT-B/16

Locked **before** training, from the only n=8 rule that fit (non-`resnet_like` jumped 6/6; linear depth/param/RF is invalid, LOO R² = −5.15, do not quote as a predictor): MSP ≥ 0.6947.

Observed MSP **0.884**, Maha 0.958, kNN 0.941. **HIT.** Report in a separate small table / paragraph. Do not add a 9th column to Kendall W. The epistemic point is the precommit, not the raw AUROC.

### 4.5 Phao B — coverage@risk, sign flip

Use the locked tables in `outputs/reports/combined_rank_utility_thesis.md`. Do not upgrade to “AUROC-winner = utility-loser.”

- Camelyon: EffB3@300 tops MSP AUROC (0.803) and is not the coverage@risk10% winner (R50 0.149 vs EffB3@300 0.107 vs EffB3@224 0.013).
- MIDOG n=3: R50 tops AUROC (0.512) and **bottoms** coverage (0.166); EffB3 tops coverage (0.712) with MSP 0.486.

Reviewer-safe sentence (already locked): *do not choose a backbone for triage from an AUROC leaderboard; measure coverage@risk on the deployment domain, because the sign of the mismatch cannot be predicted from Camelyon.*

MIDOG AUROC gaps are small; do not over-read winners. n=3 is enough for the sign-flip existence proof. Do not imply a MIDOG-8 zoo.

### 4.6 Frozen foundation models (footnote / short paragraph)

Phao A closed: GigaPath / Phikon Camelyon MSP 0.664 / 0.605, both >0.5. UNI 0.599. No invert. MIDOG-only invert footnote if mentioned. FM Maha ~0.995–1.0 is discussion support that feature>logit survives a training-paradigm change, **under cửa 2**, not a pillar.

---

## 5. Analysis / robustness appendix (failed alternatives)

This list is a contribution. Do not hide it. One subsection each, short:

| test | result | write as |
|---|---|---|
| EffB3 @224 vs @300 | jump survives (0.787) | not resolution |
| EffV2-S @224 vs @384 | Camelyon jump survives (0.778); skin kNN rank-7 dies | domain-asymmetric resolution |
| Temperature T* vs jump | r=0.268, p=0.52; Dense−R18 gap 0.308→0.308 | not calibration. T-scale is not f(MSP); do not write a monotonicity theorem |
| ID-only Maha/MSP gate | Camelyon Maha 2/8; mean AUROC 0.837 vs Maha 0.940 | not a detector; bar was Maha and ViM |
| CE+SupCon (λ=0.1, τ=0.07) | gap 0.308→0.238 (23%, bar 50%) | **not a recipe.** R18 MSP 0.574→**0.515** (worse): part of the “shrink” is the low model degrading, so the mechanism is dirty even before the 50% miss |
| ECE, BN-drift, eigenspace, n_train, CIFAR-10-C | closed earlier | appendix pointers only; CIFAR not a pillar |

Do not open center-loss, triplet, coverage-surrogate, or ensemble. Freeze.

---

## 6. Discussion

**Say what A means:** under this histopathology hospital shift, logit/probability OOD scores are backbone-dependent; feature-space scores are the stable default **when the shift looks like Camelyon**. That is a protocol warning, not a new detector.

**Skin:** feature rank-stability can hold without MSP-jump. Architecture-instability of MSP is **not** a cross-medical law. OpenMIBOOD’s feature>logit is replicated under a zoo here; our increment is the ranking movement, when it happens.

**ViT:** the pattern is not a CNN-family description of n=8. It was a prediction.

### Limitations (do not bury)

1. **iWildCam bound.** Location-shift natural images, R18/R50/DenseNet: DenseNet MSP 0.575 vs ResNet mean 0.604, jump **−0.029**. We do **not** see the Camelyon MSP-jump on this non-medical covariate shift. The phenomenon may be biomedical-specific. We did not hunt a fourth medical domain, and we do not treat the miss as a failed experiment.
2. **No mitigation.** CE+SupCon does not close the DenseNet−ResNet MSP gap; the ID-only gate is not a detector. We do not claim a fix.
3. **MIDOG is n=3**, reused as an OpenMIBOOD reference + Phao B, not a matched 8-backbone zoo.
4. **W is per-domain.** Do not average 0.694 / 0.791 / 0.857.
5. **No reader study.** Operational claim stops at coverage@risk, not at clinician agreement.
6. **Architecture is confounded with its standard training recipe (optimizer/schedule); we did not run a single-optimizer-across-backbones ablation.** Only resolution was isolated. Do not let §3.2’s “not forced-identical Adam” sit as a methods aside without this sentence here.

**Toolkit (one sentence, last):** we will release the architecture-zoo eval + coverage@risk, crediting OpenMIBOOD for MIDOG imglists, the public R50, and the 1a / cs-ID / near-OOD protocol. We will not re-sell their 24-method R50 table.

---

## 7. Conclusion

Detector choice under covariate shift is backbone-dependent on Camelyon and domain-dependent in utility (coverage). Feature-space ranks are stable when the jump is present and also when it is not (skin). A pre-registered ViT hit suggests the Camelyon logit instability is not a CNN accident. It did not appear on iWildCam. There is no new method in this paper; the protocol implication is the result.

---

## Drafting order (after this outline is approved)

1. Related work (locked paragraph + three short neighbors) — highest risk of drift.
2. Limitations paragraph (iWildCam + no-mitigation) — second-highest: do not sand off.
3. Results 4.1–4.3 tables from freeze file only.
4. Intro contributions.
5. Methods 3.1–3.3.
6. Appendix of killed tests.
7. Abstract last.

Figures (minimum): (i) Camelyon MSP vs backbone, ResNet mean + jump line; (ii) rank heatmap Camelyon vs skin; (iii) Phao B two-domain AUROC vs coverage bars. ViT as a single highlighted bar next to the jump line, caption “held-out, precommitted.”
