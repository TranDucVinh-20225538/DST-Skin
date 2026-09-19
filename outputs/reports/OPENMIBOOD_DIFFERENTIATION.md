# OpenMIBOOD / OpenOOD / OoD-Bench differentiation — locked 2026-09-18

Read before writing related work. Do not invent overlap or invent uniqueness.

Sources actually read:
- OpenMIBOOD CVPR 2025 camera-ready + supplemental Table 5
- OpenMIBOOD arXiv 2503.16247v2 (Oct 2025 corrigendum: ViM/NNGuide numbers only)
- OpenMIBOOD GitHub `remic-othr/OpenMIBOOD` (24 post-hoc methods, 3 frozen classifiers)
- OpenOOD NeurIPS 2022 D&B §4.1
- OoD-Bench CVPR 2022 abstract + intro
- NeurIPS 2026 E&D CFP + blog 2026-03-23

## Verdict (one line)

OpenMIBOOD is a **static method leaderboard on one backbone per medical domain**. It does **not** run an architecture zoo, does **not** compute Kendall's W, and does **not** test whether method ranking is stable across CNNs. The architecture-instability axis is still ours. The **real** overlap is a different sentence: they already showed **feature > logit on medical** and **method ranking does not transfer from ImageNet to medical**.

## What OpenMIBOOD actually did

| axis | OpenMIBOOD | DST-Skin |
|---|---|---|
| domains | MIDOG histopathology, PhaKIR endoscopy, OASIS-3 MRI | skin ISIC→PAD, Camelyon17 hospital-2, MIDOG (reuse) |
| methods | 24 post-hoc (MSP, EBO, MDS, KNN, ViM, ReAct, …) | 7 post-hoc (MSP, Energy, ELogitNorm, ViM, ReAct, Maha, kNN) |
| backbone per domain | **exactly one**, frozen | **zoo** (target 8 CNN families) + 3 frozen FMs |
| MIDOG classifier | ResNet-50 ImageNet, SGD, 300 ep, 50×50 crops | same public checkpoint as one baseline cell |
| PhaKIR | ResNet-18 | not used |
| OASIS-3 | R(2+1)D | not used |
| ranking object | methods, averaged AUROC across 3 MIBs (near-OOD) | methods **across backbones inside one domain** |
| rank-stability statistic | none (no Kendall W, no Spearman across arch) | Kendall's W **per domain, never pooled** |
| cross-domain rank plot | Fig. 3: method rank on MIBs vs ImageNet-1k | not our claim |
| operational metric | AUROC, FPR@95, AUPR-IN/OUT harmonic mean | AUROC **and** coverage@risk10% (OOD-triage) |
| code/checkpoints | yes (Zenodo 14982267) | planned suite; not a uniqueness claim vs them |

Supplemental Table 5 (verbatim structure): MIDOG = ResNet50; PhaKIR = ResNet18; OASIS-3 = R(2+1)D. The phrase “reuse existing model architectures” means **swap the last FC** to match class count, not sweep architectures.

GitHub: one checkpoint per MIB; scripts `eval_ood_midog.py` / `eval_ood_phakir.py` / `eval_ood_oasis3.py`. No second CNN on MIDOG.

## What they already claimed — do not re-claim as ours

1. **Natural-image OOD findings do not transfer to medical** (abstract, §5).
2. **Feature-based post-hoc > logit/probability post-hoc on these medical sets** (§5: “methods that rely on feature space information consistently outperform methods that depend solely on probabilities and logits”).
3. **Method ranking on ImageNet ≠ method ranking on medical** (Fig. 3). Closest existing “rank instability” — it is **domain** instability of a method leaderboard, architecture held fixed.
4. Classification-based scores can invert (PhaKIR EndoSeg18: OOD softmax > ID softmax). Related to our MSP-jump **phenomenon**, but they explain it as overconfidence on one dataset, not as an architecture effect.

Skin Maha rank-1 and “feature more stable than logit” **echo (2)**. Write as **replication + new axis** (architecture, plus derm/Camelyon), not as a discovery of feature>logit.

## What they did not do — this is the differentiation

- No EfficientNet / DenseNet / ConvNeXt / MobileNet / RegNet on the same ID/OOD split.
- No Kendall's W (or any concordance) of the 24-method ranking across backbones.
- No test of whether MDS/ViM stay #1 if the classifier is not ResNet-50.
- No Camelyon17 WILDS hospital shift; no dermatoscopy.
- No AUROC vs coverage@risk disagreement (they discuss AUPR class imbalance, still AUROC-family metrics).
- No resolution-controlled architecture ablation (EffB3@224 vs @300).

**Locked related-work sentence (use this, do not paraphrase into a bigger claim):**

> OpenMIBOOD (CVPR 2025) ranks 24 post-hoc detectors on three medical benchmarks with a **single frozen classifier per domain** (ResNet-50 on MIDOG, ResNet-18 on PhaKIR, R(2+1)D on OASIS-3). It shows that ImageNet method ranking does not transfer to medical data and that feature-space scores outperform logit scores under that fixed-architecture protocol. It does not measure whether the detector ranking itself is stable across architectures. We keep their MIDOG split and public ResNet-50 checkpoint as an external reference, and ask the orthogonal question: **holding the dataset fixed, how much does detector ranking move when the backbone changes?**

## Shared infrastructure — methods body, not a footnote

Put this in **methods**, first time MIDOG appears, not only in related work:

> We build on OpenMIBOOD's public MIDOG split and ResNet-50 checkpoint as an external reference.

- MIDOG 1a ID / 1b+1c cs-ID / remaining near-OOD follows OpenMIBOOD imglists.
- Public R50 AUROC ~0.59 is **their** checkpoint; in-house R50 MSP 0.512 is **ours**. Never mix. Combined thesis already locks this.
- Goal: reviewer should read reuse as field-standard, not as a scoop they found.
- Overlap a reviewer can still weaponize: “incremental OpenMIBOOD with two extra datasets.” Counter only works if the **architecture axis + utility mismatch** is the headline, not “yet another medical OOD table.”

## OpenOOD (NeurIPS 2022 D&B) and OoD-Bench (CVPR 2022)

**OpenOOD v1:** unified codebase, 30+ methods, generalized OOD taxonomy. §4.1 **fixes architecture on purpose**: LeNet on MNIST, ResNet-18 on CIFAR/TinyImageNet, ResNet-50 on ImageNet. Ranking is method ranking. OpenMIBOOD is the medical fork of this v1 protocol.

**OpenOOD v1.5 is not “still not an architecture zoo.”** On ImageNet-1K, post-hoc methods are evaluated on **ResNet-50, ViT-B/16, and Swin-T** (torchvision). Fig. 3: some methods are sensitive to architecture. Body: “different post-processor may favor different architecture” (ASH/ReAct degrade on transformers; RMDS suits them). Three models, **semantic** near/far OOD, no Kendall W, no 8-CNN family sweep, not covariate hospital shift. Closest natural-image cousin. Cite it. Do not recycle the v1 “architecture fixed” sentence as if it covered v1.5.

**OoD-Bench:** OOD **generalization** (ERM/IRM/DG), not post-hoc detection. Cite as venue precedent for “benchmark + quantify two dimensions,” not as competitor on detector ranking.

Venue-fit takeaway: CVPR/NeurIPS already accept (a) medical OOD benchmarks (OpenMIBOOD), (b) detection codebases (OpenOOD), (c) “quantify two dimensions” papers (OoD-Bench). Fit is real. Novelty is **not** “we also benchmark medical OOD.”

## Natural-image ranking vs backbone — dedicated search 2026-09-18

Question: has anyone on **natural images** (not medical) already measured OOD-detector ranking changing with backbone? Not answered by OpenOOD v1 §4.1 alone.

**Yes. Do not claim uniqueness of that observation.**

| paper | what they actually did | not what we do |
|---|---|---|
| OpenOOD v1.5 Fig. 3 | R50 vs ViT-B/16 vs Swin-T on ImageNet-1K post-hoc; architecture-favoring post-processors | 3 models; semantic OOD; qualitative; no W |
| Szyc, Walkowiak, Maciejewski, UAI 2023 (PMLR 216) | Detector **rankings change** across 3 ResNet-101/110 CIFAR variants (matched acc) and across seeds; feature methods most unstable | Reliability of SoTA ranks; not an 8-family zoo on one covariate split |
| Datko, Szyc, Walkowiak, Maciejewski, ESWA 321:132191 (2026) | Preferred post-hoc detector **per CNN/ViT line**, claimed stable across OOD sets for that model | Architecture-specific winner that transfers across OOD benchmarks; not Kendall W on one shift |
| Claros Olivares & Brockmeier, arXiv 2511.11934 (2025–26) | Scratch CNN vs fine-tuned ViT; ranking cliques of CSFs by shift severity (CIFAR / TinyImageNet); **AURC/AUGRC as primary metrics** | Two paradigms × many OOD regimes; method-selection guide; not AUROC vs coverage@risk10% sign-flip on medical covariate |

**Increment that survives:** one **medical covariate-shift** split held fixed; **8 CNN families** + precommitted ViT; Kendall W; structured MSP jump whose seed-robust core is DenseNet vs ResNet, with feature methods rank 1–2; skin non-replication; iWildCam miss on natural **covariate** shift. Phao B is not “AUROC and coverage-based ranking can disagree” (AURC/AUGRC already assume that). It is that the *sign* of AUROC vs coverage@risk10% disagreement flips, unpredictably, between Camelyon and MIDOG under the same protocol. iWildCam is the natural-image check we actually ran; OpenOOD Fig. 3 is semantic ImageNet, not a substitute.

Reviewer weapon: “OpenOOD v1.5 Fig. 3 already showed this.” Answer in related work, not for the first time in rebuttal.

## NeurIPS Evaluations & Datasets — pinned

Track rename and scope are real (blog 2026-03-23). In-scope explicitly includes: failure modes of existing evaluations, comparing evaluation designs that change conclusions, stress-testing prior evaluations, new protocols, tools, **no new model required**.

**Backup venue is NeurIPS 2027 E&D. Not 2026.**

| event | date |
|---|---|
| E&D 2026 abstract | 2026-05-04 AoE — closed |
| E&D 2026 full paper | 2026-05-06 AoE — closed |
| E&D 2026 notification | 2026-09-24 AoE |
| ICLR 2027 abstract | 2026-09-18 AoE — not a realistic target |
| ICLR 2027 paper | 2026-09-25 AoE |
| CVPR 2027 registration | 2026-11-10 AoE |
| CVPR 2027 paper | **2026-11-16 AoE — current target** |
| **NeurIPS 2027 E&D** | **CFP not out; historically ~May 2027 — pinned backup** |
| ICML 2027 | CFP not out; historically ~Jan 2027 — first post-CVPR ML venue |

E&D 2027 code policy (expect same as 2026): if the primary contribution is a reusable benchmark suite, **code must be executable at submission**, hosted, anonymized, no PI email. Croissant if new datasets. Audit-shaped papers: code encouraged, still release.

## Toolkit release

Valid **contribution bullet**, not a substitute for the finding. Do it **last**, before submit.

Credit OpenMIBOOD in README and paper for: MIDOG imglists, public R50 checkpoint, and the 1a / cs-ID / near-OOD eval protocol. Package only what is ours: architecture zoo, FID-severity, OOD-triage coverage@risk. Do not re-sell their 24-method R50 table as part of a “new” MIDOG leaderboard.

## Floor revoked — Maha>MSP cannot be a paper

OpenMIBOOD §5 already published feature>logit on medical. There is **no** workshop / CMPB-resubmit / main-track paper whose load-bearing claim is Maha>MSP.

- A or B: the paper is cửa 2 (architecture rank instability) + Phao B (AUROC↔coverage unpredictability). Skin Maha 8/8 is evidence under cửa 2, not a fallback thesis.
- C (Maha/kNN dragged): **no retreat.** Direction #1 and/or #3 become the rescue (detect + fix), or there is no paper. The old “cứt = negative Maha>MSP benchmark” frame is revoked.
- Direction #1 is still not required to clear OpenMIBOOD overlap on A/B. On C it is no longer optional.

## Implications for the 5-step list

1. Differentiation vs OpenMIBOOD: **done.** Related-work paragraph above is locked.
2. Direction #1: A/B → still no unless we want a method-paper; **C → mandatory rescue with #1 and/or #3.**
3. Direction #2 + #4: cheap add-ons; #2 is also the E&D-shaped protocol.
4. Toolkit: last, with OpenMIBOOD credit as above.
5. Venue: CVPR 2027 main = 2026-11-16. Backup of E&D **shape** = **NeurIPS 2027 E&D**. No Maha>MSP floor at any remaining venue.

## Forbidden sentences

- “First medical OOD benchmark”
- “First to show feature-based OOD beats MSP in medical imaging”
- Any abstract/contribution bullet whose independent claim is Maha>MSP
- “OpenMIBOOD already swept architectures” (false)
- “NeurIPS 2026 E&D is our backup” (false; use 2027)
- Quoting their MDSEns as comparable without their own caveat (validation uses OOD labels)
- “Nobody has measured detector ranking vs backbone on natural images” (false; OpenOOD v1.5 Fig. 3, Szyc et al. UAI 2023, Datko et al. ESWA 2026)
- “OpenOOD v1.5 still fixes architecture” / “v1.5 is still not an architecture zoo” (false)
