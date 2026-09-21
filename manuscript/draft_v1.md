# Detector ranking is not architecture-invariant under medical covariate shift

**CVPR 2027 LaTeX (review PDF):** `manuscript/cvpr2027/main.tex` → `main.pdf`. Supplement: `suppl.tex`. Numbers from `outputs/reports/OFFICIAL_W_FREEZE.txt`. Official Kendall $W$ is never recomputed in this file.

---

## Abstract

Post-hoc out-of-distribution (OOD) detectors are almost always ranked on a **single** backbone. Holding a medical **covariate** split fixed, that ranking is not architecture-invariant. On Camelyon17 hospital-2 (WILDS), Kendall’s $W=0.694$ across eight CNN families and seven scores (seed 42). MSP AUROC spans $0.515$ (ResNet-50) to $0.883$ (DenseNet-121); the DenseNet jump versus the ResNet mean is seed-robust ($0.265\pm 0.065$, $5/5$ seeds $\ge 0.15$). ConvNeXt-Tiny and EfficientNetV2-S@224 hold on $4/5$ seeds; MobileNetV3 and RegNetY on $3/5$; EfficientNet-B3’s seed-42 jump is not seed-stable ($2/5$). Feature-space scores (Mahalanobis, $k$NN) occupy ranks 1–2 on every official backbone. The same eight-family protocol on skin ISIC→PAD does **not** reproduce the MSP jump ($W=0.791$; Maha rank-1 on $8/8$). AUROC rank and OOD-triage coverage@risk$10\%$ disagree on Camelyon and on MIDOG, and the **sign** of that disagreement flips. A pre-registered ViT-B/16 Camelyon MSP of $0.884$ met a locked threshold of $0.6947$. We introduce no new detector. Do not copy an OOD ranking across backbones, and do not pick a backbone for triage from an AUROC table.

---

## 1. Introduction

On the Camelyon17 hospital-2 split, with the same seven post-hoc scores, ResNet-18/50 MSP AUROC is $0.574$/$0.515$ while DenseNet-121 is $0.883$ and EfficientNet-B3 (seed 42) is $0.803$. A practitioner who copies an OpenMIBOOD-style ResNet-50 recipe on this shift and one who uses DenseNet are not looking at the same detector ranking.

That observation is **not** a claim that detector ranking was previously assumed architecture-invariant. On natural-image **semantic** OOD, ranking already moves with architecture (OpenOOD v1.5; Szyc et al.; Datko et al.; Claros Olivares & Brockmeier). OpenMIBOOD (CVPR 2025) already showed that ImageNet method ranking does not transfer to medical data and that feature-space scores outperform logit scores under a **frozen** classifier per domain. Our question is orthogonal: **holding a medical covariate-shift dataset fixed, how much does detector ranking move when the backbone changes?**

We measure that movement at zoo scale on Camelyon17, show that it does not replicate as an MSP jump on skin ISIC→PAD, and show that AUROC versus operational coverage@risk$10\%$ disagrees with a **domain-dependent sign**. We pre-registered a ViT-B/16 MSP threshold before training that model. We do not claim a new detector, a Maha$>$MSP discovery, or a two-domain MSP-jump law.

**Contributions.**

1. **Camelyon-17, zoo-scale concordance.** Official Kendall $W=0.694$ ($n=8$, 7 methods, seed 42). The seed-robust MSP jump is DenseNet-121 versus ResNet ($0.265\pm 0.065$, $5/5$). ConvNeXt-Tiny and EfficientNetV2-S@224 are $4/5$; MobileNetV3 and RegNetY are $3/5$; EfficientNet-B3 is $2/5$. Matching the four non-Adam zoo members to ResNet-Adam (M1) does not remove those seed-42 jumps. Feature-space scores stay in the top two ranks. This is not a claim that architecture-sensitive ranking is new.
2. **Domain asymmetry.** Skin ISIC→PAD ($n=8$, $W=0.791$): Maha rank-1 and $k$NN rank-2 on all 8; MSP stays in a narrow band without a Camelyon-style ResNet-vs-rest jump. Not a second scenario A.
3. **Utility mismatch is unpredictable.** Not that AUROC and coverage-based ranking can disagree (AURC/AUGRC already assume that). The *sign* of AUROC versus coverage@risk$10\%$ disagreement flips between Camelyon and MIDOG, with no rule to predict which backbone wins. Do not pick a backbone for triage from an AUROC leaderboard.
4. **A locked prediction.** Before training: ViT-B/16 Camelyon MSP $\ge 0.6947$. Observed $0.884$. Held out of the 8-CNN $W$ on purpose.

---

## 2. Related work

OpenMIBOOD (CVPR 2025) ranks 24 post-hoc detectors on three medical benchmarks with a **single frozen classifier per domain** (ResNet-50 on MIDOG, ResNet-18 on PhaKIR, R(2+1)D on OASIS-3). It shows that ImageNet method ranking does not transfer to medical data and that feature-space scores outperform logit scores under that fixed-architecture protocol. It does not measure whether the detector ranking itself is stable across architectures. We keep their MIDOG split and public ResNet-50 checkpoint as an external reference, and ask the orthogonal question: **holding the dataset fixed, how much does detector ranking move when the backbone changes?**

OpenOOD v1 (Yang et al., NeurIPS 2022 Datasets & Benchmarks) **fixes architecture on purpose** (LeNet / ResNet-18 / ResNet-50). That design choice is about v1, not about the later literature.

Architecture-sensitive post-hoc ranking **has** been measured on natural images. OpenOOD v1.5 (Zhang et al.) evaluates ImageNet-1K post-hoc methods on ResNet-50, ViT-B/16, and Swin-T (torchvision checkpoints). Figure 3 and the accompanying text report that “different post-processor may favor different architecture”: ASH/ReAct degrade on transformers, while RMDS suits them. That study uses **three** models, **semantic** near/far OOD, and a qualitative comparison. It does not compute Kendall’s $W$, does not sweep eight CNN families, and is not a covariate hospital shift. It is the closest natural-image cousin; we do not treat v1.5 as an architecture-fixed protocol.

Szyc, Walkowiak & Maciejewski (UAI 2023) show that detector **rankings change** across three ResNet-101/110 CIFAR implementation variants with matched closed-set accuracy, and across random seeds, with feature-space methods especially unstable. That is a reliability result about uncontrolled experimental details on **CIFAR semantic OOD**, not an eight-family zoo that holds one medical covariate split fixed.

Datko, Szyc, Walkowiak & Maciejewski (Expert Systems with Applications, 2026) report a preferred post-hoc detector **per CNN/ViT line**, claimed to be stable across OOD benchmarks for that model. The framing is an architecture-specific winner that **transfers** across natural-image OOD sets—the opposite of a single-split concordance statistic.

Claros Olivares & Brockmeier (arXiv 2511.11934) compare a scratch CNN to a fine-tuned ViT and organize confidence-score rankings into cliques by shift severity on CIFAR / TinyImageNet. That is two representation paradigms times many OOD regimes, as a method-selection guide. It is not eight CNN families on one medical covariate split. Their released results rank CSFs with AURC/AUGRC; they *do* use risk–coverage curves as the ranking object. They do not report an AUROC-rank versus coverage@risk$10\%$ disagreement whose sign flips across medical covariate domains.

What remains after these papers is not the existence of architecture-sensitive ranking. That existence is published on natural-image semantic OOD. This paper measures **one medical covariate-shift split, held fixed**; an **eight CNN-family zoo** plus a **precommitted** ViT; **Kendall’s $W$ per domain**; a **structured MSP jump** whose seed-robust core is DenseNet versus ResNet, with feature methods in ranks 1–2; **skin non-replication** of the MSP jump; an **iWildCam miss** on natural-image **covariate** shift; and the coverage sign-flip below. That sign-flip is not the observation that AUROC and a coverage-based ranking can disagree — AURC/AUGRC exist because that disagreement is already assumed. It is that the *sign* of the AUROC versus coverage@risk$10\%$ disagreement flips, with no rule to predict which backbone wins, between two medical covariate-shift domains run under the same protocol (Camelyon vs MIDOG). iWildCam is the natural-image covariate check we actually ran; OpenOOD v1.5 Fig. 3 is semantic ImageNet, not a substitute.

OoD-Bench (CVPR 2022) quantifies OOD **generalization** (ERM/IRM/domain generalization), not post-hoc detection. We cite it only as venue precedent for a two-axis evaluation paper. WILDS Camelyon17 and iWildCam supply the hospital / camera-trap covariate splits; iWildCam is a negative bound in this paper, not a third main domain.

---

## 3. Setup

### 3.1 Shifts

| Domain | ID | OOD | Role in this paper |
|---|---|---|---|
| Camelyon17 WILDS | hospitals 0, 3, 4 | hospital 2 (`test`) | founding cell; official $W$, $n=8$ |
| Skin ISIC→PAD | ISIC | PAD-UFES-20 | feature rank-stability without MSP-jump |
| MIDOG | OpenMIBOOD 1a | 1b+1c (+ remaining near-OOD in their protocol) | coverage sign-flip + external R50 reference; official $W$ is **$n=3$** |

We build on OpenMIBOOD's public MIDOG split and ResNet-50 checkpoint as an external reference. Public R50 AUROC $\sim 0.59$ is **theirs**. In-house R50 MSP $0.512$ is **ours**. We never mix the two.

### 3.2 Backbones (official $W$)

Eight CNN-family models, ImageNet initialization, 10-epoch fine-tune, seed 42, train-frac $1.0$ on Camelyon. Standard fine-tune recipes (Adam / AdamW / SGD, learning rates, weight decay) follow `src/models/cnn_family.py` and are **not** forced-identical Adam. That confound is measured in §4.3 (M1/M2) and stated in Limitations. Only **resolution** was isolated in a controlled pair (EffB3 @224 vs @300; EffV2-S @224 vs @384).

Official EffV2-S is **224** on Camelyon and on skin (native 384 dropped). Official EffB3 Camelyon remains **300**; EffB3@224 is an appendix control (MSP $0.787$). ViT-B/16 @224 is held out of $W$ and reported separately. Not Swin-T.

### 3.3 Scores and statistics

Seven post-hoc scores: MSP, Energy, ELogitNorm, ViM, ReAct, Mahalanobis (Ledoit–Wolf), $k$NN. Primary concordance: **Kendall’s $W$ per domain, never pooled**. Jump: $\mathrm{MSP}-\mathrm{mean}(\mathrm{R18},\mathrm{R50})\ge 0.15$. Feature drag: Maha or $k$NN on a new backbone $<\mathrm{median}(\mathrm{R18},\mathrm{R50},\mathrm{EffB3})-0.10$. Code: `scripts/rebuild_architecture_invariance.py`. Official Camelyon ResNet mean MSP $=0.544654$; jump iff MSP $\ge 0.695$ on that split.

Coverage@risk$10\%$ is OOD-triage (`pathology_risk_coverage.py`): rank OOD samples by MSP (higher $=$ more ID-like), take the largest coverage whose classifier risk is $\le 10\%$. ID-selective Camelyon is vacuous (ID-val acc $\approx 99.5\%$) and is not cited.

### 3.4 Failed alternatives (pointer)

We tested whether the Camelyon MSP pattern is an artifact of input resolution, temperature scaling, ID-only detector **selection**, a CE+SupCon recipe, or train-time HED stain jitter. None of these is a clean account or a clean mitigation (§5 / appendix). Temperature scaling is **not** treated as a monotone of MSP AUROC.

---

## 4. Results

### 4.1 Camelyon-17 — official table (seed 42) and five-seed MSP

Official Kendall $W=\mathbf{0.694}$ ($n=8$, 7 methods). This is the only Camelyon number called “official $W$.” Leave-one-backbone recomputation of $W$ (dropping one of the eight) ranges $0.676$–$0.738$; that is **structural sensitivity**, not a $95\%$ CI.

**Figure 1 (to insert).** Camelyon hospital-2 MSP AUROC by backbone, seed 42. Dashed lines at the ResNet mean ($0.545$) and the jump threshold ($0.695$). ViT-B/16 is held out of $W$ and plotted separately.

**Table 1.** Camelyon hospital-2 AUROC, **seed 42** (official $W$). Jump iff MSP $\ge 0.6947$ versus frozen ResNet mean $0.544654$ (R18 $0.574$, R50 $0.515$). Maha/$k$NN ranks among the seven scores. MSP range on this table is R50$\to$DenseNet.

| Backbone | input | Maha (rank) | $k$NN (rank) | MSP | seed-42 jump |
|---|---|---|---|---|---|
| ResNet-18 | 224 | 0.956 (1) | 0.936 (2) | 0.574 | no |
| ResNet-50 | 224 | 0.896 (1) | 0.866 (2) | 0.515 | no |
| DenseNet-121 | 224 | 0.988 (1) | 0.973 (2) | 0.883 | yes |
| ConvNeXt-Tiny | 224 | 0.928 (1) | 0.916 (2) | 0.846 | yes |
| MobileNetV3-L | 224 | 0.870 (2) | 0.904 (1) | 0.771 | yes |
| RegNetY-3.2GF | 224 | 0.964 (1) | 0.956 (2) | 0.784 | yes |
| EfficientNet-B3 | 300 | 0.885 (2) | 0.934 (1) | 0.803 | yes |
| EfficientNetV2-S | **224** | 0.880 (1) | 0.856 (2) | 0.778 | yes |

Feature methods occupy ranks 1–2 on every official backbone (Maha \#1 on $6/8$; $k$NN \#1 on EffB3 and MobileNet). No backbone drags Maha/$k$NN past the $0.10$ feature-drag threshold. On this table, MSP spans $0.515$–$0.883$; Maha $0.870$–$0.988$; $k$NN $0.856$–$0.973$. Incomplete concordance ($W=0.694$) is the logit ranks moving.

**Table 2.** Same eight Camelyon hospital-2 MSP cells on all five seeds $\{42,43,44,45,46\}$. Jump $=$ MSP $-$ that seed’s ResNet mean; bar $0.15$. The last column is how many of those five seeds clear the bar. Official $W$ is seed 42 only. Per-seed jumps are Table A.5.

| Backbone | 42 | 43 | 44 | 45 | 46 | jump mean$\pm$sd | $\ge 0.15$ |
|---|---:|---:|---:|---:|---:|---:|---|
| ResNet-18 | 0.574 | 0.712 | 0.601 | 0.524 | 0.581 | — | — |
| ResNet-50 | 0.515 | 0.621 | 0.619 | 0.515 | 0.661 | — | — |
| ResNet mean | 0.545 | 0.667 | 0.610 | 0.520 | 0.621 | — | — |
| DenseNet-121 | 0.883 | 0.865 | 0.868 | 0.845 | 0.826 | $0.265\pm 0.065$ | $5/5$ |
| ConvNeXt-Tiny | 0.846 | 0.799 | 0.803 | 0.764 | 0.785 | $0.207\pm 0.067$ | $4/5$ |
| MobileNetV3-L | 0.771 | 0.728 | 0.815 | 0.780 | 0.745 | $0.175\pm 0.081$ | $3/5$ |
| RegNetY-3.2GF | 0.784 | 0.778 | 0.836 | 0.830 | 0.724 | $0.198\pm 0.089$ | $3/5$ |
| EfficientNet-B3 | 0.803 | 0.731 | 0.643 | 0.818 | 0.678 | $0.142\pm 0.125$ | $2/5$ |
| EfficientNetV2-S@224 | 0.778 | 0.744 | 0.808 | 0.846 | 0.835 | $0.210\pm 0.089$ | $4/5$ |

DenseNet-121 is the confirmed seed-robust jump ($5/5$). ConvNeXt-Tiny and EfficientNetV2-S@224 are not seed-42 flukes ($4/5$); we do not promote them to $5/5$. MobileNetV3 and RegNetY are $3/5$. EfficientNet-B3 is $2/5$: its seed-42 jump ($+0.258$) is a favorable draw relative to $0.142\pm 0.125$. The seed-42 matrix still shows a jump without feature drag. The **prose** claim is narrower than “$6/6$ non-ResNet, confirmed.” M1 (seed 42) still shows ConvNeXt, MobileNet, and RegNet jump under ResNet-Adam; do not read M1 as a multi-seed result.

### 4.2 Skin ISIC→PAD — not a second A

Official $W=\mathbf{0.791}$ ($n=8$). Maha is rank-1 and $k$NN rank-2 on **all 8**. MSP sits in a narrow band ($0.68$–$0.74$). There is no Camelyon-style ResNet-vs-rest MSP jump. EffV2-S@224 MSP $0.737$; the native-384 cell that put $k$NN at rank 7 (MSP $0.823$) is a resolution artifact and is not in official $W$.

**Figure 2 (to insert).** Rank heatmap, Camelyon vs skin: seven methods $\times$ eight backbones.

Skin is **rank-stability of feature scores** under a zoo: a replication of OpenMIBOOD’s feature$>$logit pattern **across architectures**, not an MSP-jump and not a second A.

### 4.3 Robustness: resolution, then optimizer (M1 and M2)

**Resolution is not the Camelyon DenseNet/seed-42 pattern.** EffB3 Camelyon: $0.803$ @300 $\to$ $0.787$ @224 (jump vs the official ResNet mean survives). EffV2-S Camelyon: $0.726$ @384 $\to$ **$0.778$ @224** (the official cell; jump is not a 384 artifact). On skin, the EffV2-S @384 $k$NN collapse **is** a resolution artifact and disappears at 224. These two facts are not a slogan that “resolution never matters.”

**M1 — jumpers trained with ResNet Adam** ($\mathrm{lr}=10^{-4}$, $\mathrm{wd}=10^{-4}$, batch $64$). The four official zoo members whose **standard** recipe was not that Adam (ConvNeXt, MobileNetV3, RegNet, EffV2-S@224) still jump versus the **frozen** official ResNet mean $0.545$:

| Backbone | M1 MSP | jump vs $0.545$ | id-val acc |
|---|---|---|---|
| ConvNeXt-Tiny | 0.820 | $+0.275$ | 0.9944 |
| MobileNetV3-L | 0.737 | $+0.193$ | 0.9949 |
| RegNetY-3.2GF | 0.805 | $+0.261$ | 0.9966 |
| EfficientNetV2-S@224 | 0.873 | $+0.328$ | 0.9961 |

$4/4$ clean hits. DenseNet and EffB3 already shared ResNet-Adam in the official zoo, so they were not retrained here. Recipe mismatch does **not** explain the seed-42 jumps of those four members. The seed-42 “families beyond EfficientNet” pattern is therefore not an optimizer-only artifact of AdamW/SGD. M1 is seed 42 only; the five-seed MSP matrix is Table 2, not this table.

**M2 — ResNets trained with ConvNeXt AdamW** ($\mathrm{wd}=0.05$). ResNet-18 MSP **$0.773$** (enters the $0.695$ band; id-val $0.9957$). ResNet-50 MSP $0.581$ (stays out; id-val $0.9958$). ResNet **can** jump under a non-default recipe. Architecture is not a clean variable, independent of optimizer.

These two results point in opposite directions and we report both.

### 4.4 Held-out prediction — ViT-B/16

Locked **before** training, from the seed-42 zoo rule that non-`resnet_like` cells jumped (linear depth/parameter/receptive-field predictors are invalid on this $n=8$ and are not quoted; LOO $R^2=-5.15$ is not used as a predictor): MSP $\ge 0.6947$. Observed MSP **$0.884$**, Maha $0.958$, $k$NN $0.941$. The prediction holds. Not a ninth $W$ column. The point is the precommit.

### 4.5 Coverage@risk — the sign flips

Protocol: OOD-triage coverage@risk$10\%$ with MSP. Not ID-selective coverage.

**Camelyon (hospital-2), $n=4$ in the locked utility table.** MSP AUROC order: EffB3@300 ($0.803$) $>$ EffB3@224 ($0.787$) $>$ ResNet-18 ($0.574$) $>$ ResNet-50 ($0.515$). Coverage@risk$10\%$ order: ResNet-50 ($0.149$) $>$ ResNet-18 ($0.145$) $>$ EffB3@300 ($0.107$) $>$ EffB3@224 ($0.013$). The AUROC winner is not the coverage winner.

**MIDOG, official $n=3$ (in-house).** MSP AUROC: ResNet-50 $0.512$, EffB3 $0.486$, ResNet-18 $0.438$ (gaps are small; do not over-read winners). Coverage@risk$10\%$: EffB3 $0.712$, ResNet-18 $0.442$, ResNet-50 $0.166$. ResNet-50 is AUROC-highest and coverage-lowest. The permutation is **not** the Camelyon permutation.

**Figure 3 (to insert).** Two-domain grouped bars: MSP AUROC vs coverage@risk$10\%$ for R18 / R50 / EffB3 on Camelyon and on MIDOG.

We do not write “AUROC-winner $=$ utility-loser” as a law, and we do not claim to have discovered that AUROC and coverage-based ranking can disagree — AURC/AUGRC exist because that is already assumed. The increment is narrower: the *sign* of the AUROC versus coverage@risk$10\%$ disagreement flips between two medical covariate-shift domains under the same protocol (Camelyon vs MIDOG), with no rule to predict which backbone wins. Do not choose a backbone for triage from an AUROC leaderboard; measure coverage@risk on the deployment domain. Official numbers use MIDOG $n=3$ and the locked Camelyon $n=4$ table. The Camelyon $n=8$ family span still **disagrees** (DenseNet AUROC $0.883$ / coverage $0.078$; MobileNet coverage $0.904$); that is the same sign-flip claim, not a law, and not a new official $W$.

### 4.6 Frozen foundation models (short)

We do not claim a foundation-model invert on Camelyon. Linear-probe MSP: GigaPath $0.664$, Phikon $0.605$, UNI $0.599$, all $>0.5$. MIDOG UNI / GigaPath MSP $0.445$ / $0.470$ in the sign audit (below $0.5$) is a **MIDOG-only** footnote, not a Camelyon invert. FM Maha $\sim 0.995$–$1.0$ is discussion support that feature$>$logit can survive a training-paradigm change, **under the ranking-stability question**, not a pillar.

---

## 5. Analysis: failed alternatives

Each of the following was a precommitted test. None is hidden.

**Resolution** (repeated from §4.3). EffB3@224 MSP $0.787$; EffV2-S@224 MSP $0.778$. Not a pixel-size explanation of the Camelyon seed-42 table.

**Temperature $T^\ast$.** Correlation of $T^\ast$ with the MSP jump: $r=0.268$, $p=0.52$. DenseNet$-$ResNet-18 MSP gap $0.308\to 0.308$ after $T^\ast$. EffV2 native-384 MSP under $T$-scale: $0.726\to 0.744$. Calibration/temperature is not the mechanism. We do **not** write a monotonicity theorem relating temperature scaling to MSP AUROC.

**ID-only Maha/MSP gate.** Selecting a score from ID-only statistics is a **selector**, not a detector. On Camelyon it picks Maha on $2/8$ backbones; mean AUROC of the gated score $0.837$ versus always-Maha $0.940$. The bar was matching Maha and ViM. It does not.

**CE+SupCon** ($\lambda=0.1$, $\tau=0.07$), ResNet-18 $+$ DenseNet. Gap $0.308\to 0.238$ ($23\%$; precommit bar was $50\%$ / $\le 0.154$). ResNet-18 MSP $0.574\to\mathbf{0.515}$ (worse; this is the SupCon cell, not official ResNet-50 MSP $0.515$). Part of the shrink is the low model degrading. Dirty; not a recipe.

Stain jitter, MIDOG $n=8$, and shift-type (run; no law) are appendix (A.2–A.4). We do not open center-loss, triplet, a coverage surrogate, or an ensemble.

---

## 6. Discussion

Under this histopathology **hospital** shift, logit/probability OOD scores move with the backbone: seed-robust for DenseNet ($5/5$), present on most seeds for ConvNeXt and EffV2-S@224 ($4/5$), weaker for MobileNet and RegNet ($3/5$), and seed-fragile for EfficientNet-B3 ($2/5$). Feature-space scores are the stable default **when the shift looks like Camelyon**. That is a protocol warning, not a new detector.

On skin, feature rank-stability holds **without** an MSP jump. Architecture-instability of MSP is not a cross-medical law. OpenMIBOOD’s feature$>$logit finding is replicated here under a zoo; the increment is ranking movement, **when it happens**, plus the operational AUROC$\leftrightarrow$coverage mismatch.

ViT-B/16 was a locked prediction, not a post-hoc ninth CNN. Matched-recipe M1 and M2 disagree about how cleanly that movement is “architecture” versus optimizer (§4.3); both stand.

### Limitations

1. **iWildCam bound.** Location-shift natural images. Locked 3-backbone cell: DenseNet MSP $0.575$ vs ResNet mean $0.604$, jump $-0.029$. Extra five CNNs vs that domain’s ResNet mean: ConvNeXt $0.601$, MobileNet $0.558$, RegNet $0.580$, EffB3 $0.577$, EffV2-S $0.650$; **$0/5$ jump**. We do **not** see the Camelyon MSP-jump on this non-medical covariate shift. The phenomenon may be biomedical-specific. We did not hunt a fourth medical domain, do not pool iWildCam with Camelyon $W$, and do not treat the miss as a failed experiment.
2. **No clean mitigation.** CE+SupCon is dirty (ResNet-18 MSP worse). HED stain jitter is dirty (DenseNet MSP $0.780<0.80$ precommit floor). The ID-only gate is a selector, not a detector. We do not claim a fix.
3. **MIDOG official $W$ is $n=3$** (in-house R18/R50/EffB3, $W=0.857$), reused as an OpenMIBOOD reference $+$ coverage sign-flip. An in-house $n=8$ zoo (appendix) shows no MSP jump and does **not** replace official $W$.
4. **$W$ is per-domain.** Do not average $0.694$ / $0.791$ / $0.857$.
5. **No reader study.** The operational claim stops at coverage@risk, not clinician agreement.
6. **Architecture is confounded with its standard training recipe** (optimizer / weight decay / batch size). M1: the four recipe-mismatched jumpers still jump under ResNet-Adam (seed 42). M2: ResNet-18 enters the MSP band under ConvNeXt-AdamW. Both are reported. **$5/5$ seed-robustness is DenseNet only.** ConvNeXt and EffV2-S@224 are $4/5$; MobileNet and RegNet are $3/5$; EffB3 is $2/5$. Official $W$ stays seed 42.

We did not isolate which component of the hospital shift (staining, scanner, tissue prep) drives the jump; this is left to future work.

**Toolkit (one sentence).** We will release the architecture-zoo evaluation and coverage@risk code, crediting OpenMIBOOD for MIDOG imglists, the public ResNet-50 checkpoint, and the 1a / cs-ID / near-OOD protocol. We will not re-sell their 24-method R50 table.

---

## 7. Conclusion

Under Camelyon hospital shift, post-hoc detector choice is backbone-dependent: a seed-robust DenseNet MSP jump, weaker but directionally present jumps on ConvNeXt/EffV2 ($4/5$) and MobileNet/RegNet ($3/5$), and a seed-fragile EfficientNet-B3 cell. Feature-space ranks stay in the top two whether that jump is present (Camelyon) or absent (skin). A pre-registered ViT hit shows the Camelyon logit instability is not only a CNN-zoo accident; it did not appear on iWildCam. AUROC and coverage@risk can rank backbones in opposite orders, and which way the disagreement points depends on the domain. There is no new method in this paper. The result is the protocol: do not copy an OOD ranking across backbones, and do not pick a backbone for triage from an AUROC table.

---

## Appendix

### A.1 Leave-one-backbone $W$ (not a CI)

Official Camelyon $W=0.694$ ($n=8$). Dropping one backbone and recomputing $W$ on the remaining seven yields $0.676$–$0.738$ (largest increase when dropping ConvNeXt; largest decrease when dropping EffV2-S or RegNet). This is jackknife **structural** sensitivity of the concordance statistic. It is not a seed confidence interval and is not written as “$W=0.694$, $95\%$ CI $[\cdot]$.”

### A.2 HED stain jitter (killed)

Train-only HED stain jitter ($\sigma=0.20$, $p=1.0$) on ResNet-18, ResNet-50, DenseNet-121, same Adam recipe as official, seed 42. Precommit dirty-guard: DenseNet hospital-2 MSP $<0.80$ $\Rightarrow$ dirty.

Hospital-2 MSP: ResNet-18 $0.772$ (enters the $0.695$ band), ResNet-50 $0.593$, DenseNet $0.780$ (**below $0.80$**). Gap DenseNet$-$ResNet-18 $=0.008$. id-val acc $\ge 0.994$ on all three. Coverage@risk$10\%$ saturates at $1.0$ for these three (not worse than official ResNet-18 $0.145$). Held-out hospital-1 MSP: ResNet-18 $0.765$, ResNet-50 $0.685$, DenseNet $0.796$.

**Dirty.** Part of the apparent gap shrink is DenseNet falling ($0.883\to 0.780$) while ResNet-18 rises. Same class as CE+SupCon. Appendix kill, not an open mitigation. Not expanded to eight CNNs. No coverage-surrogate follow-up.

### A.3 MIDOG in-house zoo $n=8$ (not official $W$)

Official MIDOG $W$ remains **$n=3$**, $0.857$. Five additional in-house CNNs (DenseNet, ConvNeXt, MobileNet, RegNet, EffV2-S@224) plus the original three, same 1a vs 1b+1c protocol, **not** mixed with public OpenMIBOOD R50 $\sim 0.59$:

| Backbone | MSP | Maha |
|---|---|---|
| ResNet-18 | 0.438 | 0.733 |
| ResNet-50 | 0.512 | 0.839 |
| DenseNet-121 | 0.447 | 0.910 |
| ConvNeXt-Tiny | 0.521 | 0.801 |
| MobileNetV3-L | 0.469 | 0.714 |
| RegNetY-3.2GF | 0.553 | 0.944 |
| EfficientNet-B3 | 0.486 | 0.673 |
| EfficientNetV2-S@224 | 0.493 | 0.581 |

MSP $0.44$–$0.55$ on all eight; **$0/6$ non-ResNet jumps** versus that domain’s ResNet mean. Maha remains high on most cells. Consistent with official $n=3$: no Camelyon-style MSP jump. A **domain-scope bound**, not a new main finding, and **not** a replacement of official $W$.

### A.4 Shift-type (run; not official $W$)

Same eight Camelyon seed-42 classifiers; ID $=$ Camelyon `id_val`. Arm C $=$ hospital-2 (frozen). Arm T $=$ MIDOG 1a (task-shift H\&E). Arm F $=$ CIFAR-10 test (far-OOD). Extract only; T/F CSVs are **not** mixed into `rebuild_architecture_invariance.py`.

| Arm | non-ResNet jump | readout |
|---|---|---|
| C covariate | $6/6$ | seed-42 count, not a multi-seed block |
| T task-shift | **$0/6$** | jump disappears |
| F far-OOD | $2/6$ | mixed |

Joint: **no law.** Per-arm Kendall $W$ (8 backbones $\times$ 7 methods, never pooled, not official $W$): C $0.731$, T $0.692$, F $0.846$. We still do not isolate staining vs scanner vs tissue prep inside the hospital shift; that remains future work.

### A.5 Five-seed jumps (companion to Table 2)

Jump $=$ MSP $-$ that seed’s ResNet mean. Threshold $0.15$. Official $W$ is seed 42 only. MSP matrix is Table 2.

| seed | R mean | DenseNet | ConvNeXt | MobileNet | RegNet | EffB3 | EffV2@224 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.545 | $+0.338$ | $+0.301$ | $+0.226$ | $+0.239$ | $+0.258$ | $+0.234$ |
| 43 | 0.667 | $+0.199$ | $+0.133$ | $+0.061$ | $+0.112$ | $+0.064$ | $+0.077$ |
| 44 | 0.610 | $+0.258$ | $+0.193$ | $+0.205$ | $+0.226$ | $+0.033$ | $+0.198$ |
| 45 | 0.520 | $+0.325$ | $+0.244$ | $+0.260$ | $+0.310$ | $+0.298$ | $+0.327$ |
| 46 | 0.621 | $+0.205$ | $+0.164$ | $+0.124$ | $+0.103$ | $+0.057$ | $+0.214$ |

### A.6 Other closed checks

ECE, BN-drift, eigenspace summaries, $n_{\mathrm{train}}$ sweeps, and CIFAR-10-C were closed earlier and are not pillars. CIFAR semantic/far-OOD is not a main-domain result.

---

## Numbers lock (do not edit in prose without the freeze file)

| Quantity | Value | Source |
|---|---|---|
| Official $W$ Camelyon | $0.694$ ($n=8$) | `OFFICIAL_W_FREEZE.txt` |
| Official $W$ skin | $0.791$ ($n=8$) | same |
| Official $W$ MIDOG | $0.857$ ($n=3$) | same |
| Camelyon ResNet mean MSP | $0.544654$ | same |
| DenseNet 5-seed jump | $0.265\pm 0.065$ ($5/5$) | `camelyon_seed_jump.txt` |
| ConvNeXt 5-seed jump | $0.207\pm 0.067$ ($4/5$) | same |
| EffV2@224 5-seed jump | $0.210\pm 0.089$ ($4/5$) | same |
| MobileNet 5-seed jump | $0.175\pm 0.081$ ($3/5$) | same |
| RegNet 5-seed jump | $0.198\pm 0.089$ ($3/5$) | same |
| EffB3 5-seed jump | $0.142\pm 0.125$ ($2/5$) | same |
| Shift-type hits | C $6/6$, T $0/6$, F $2/6$ | `camelyon_shift_type.txt` |
| iWildCam extra | $0/5$ jump | `iwildcam_jump.txt` |
| M1 MSPs | $0.820/0.737/0.805/0.873$ | `camelyon_matched_recipe.txt` |
| M2 R18 / R50 MSP | $0.773$ / $0.581$ | same |
| HED stain R18 / Dense MSP | $0.772$ / $0.780$ | `camelyon_stain_cov.txt` |
| ViT MSP | $0.884$ | freeze |
| iWildCam jump | $-0.029$ | freeze |
| In-house MIDOG R50 MSP | $0.512$ | never mix with public $\sim 0.59$ |
