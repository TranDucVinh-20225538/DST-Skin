# Detector ranking is not architecture-invariant under covariate shift

**Draft — CVPR 2027** (paper due 2026-11-16 AoE).

Numbers **frozen 2026-09-18**. Official Kendall W rebuilt without native-384 EffV2.
Canonical outline for section review: `manuscript/cvpr2027_outline.md`.
Freeze file: `outputs/reports/OFFICIAL_W_FREEZE.txt`.

Do not load-bear on Maha>MSP: OpenMIBOOD (CVPR 2025) already showed feature-space post-hoc beats logit methods on medical imaging with a **frozen** classifier. Our question is orthogonal: **holding the dataset fixed, how much does detector ranking move when the backbone changes?**

## Locked claims (do not drift)

1. **Cửa 2, Camelyon-17, official scenario A.** n=8, Kendall W = **0.694**. MSP jump ≥0.15 vs ResNet mean (0.545) on 6/8 CNN-family backbones, including non-EfficientNet families. EffV2-S@224 MSP **0.778** (+0.233) — jump is not a 384 artifact. Maha/kNN ranks 1–2 on every backbone.
2. **Skin ISIC→PAD does not replicate the MSP jump** (MSP 0.68–0.74). Official W = **0.791**. Maha rank 1 and kNN rank 2 on 8/8. Write skin as rank-stability of feature methods, not as a second A.
3. **Phao A closed.** Frozen UNI / GigaPath / Phikon: Camelyon MSP 0.599 / 0.664 / 0.605, all >0.5 (no invert). MIDOG-only footnote. FM Maha 0.995–1.0 is discussion support for cửa 2, not a pillar.
4. **Phao B** (AUROC vs coverage@risk10% sign flip across Camelyon vs MIDOG) stays. Do not write “AUROC-winner = utility-loser.”
5. **ViT-B/16 precommit HIT** (MSP 0.884 ≥ 0.6947). Held out of official W.
6. **iWildCam MISS** (jump −0.029) = biomedical bound. Limitations, not a fourth domain.
7. **No method.** Job B dead (gap 0.308→0.238; R18 MSP 0.574→0.515). ID-gate not a detector. Freeze: no more jobs.

**Forbidden:** pooled Kendall W; quoting unofficial mixed-res W=0.666; native-384 EffV2 in the main table; “MSP-jump is a general law.”

## Official Camelyon-8 (EffV2 = 224; EffB3 = 300)

ResNet mean MSP = 0.545. Jump threshold 0.15.

| Backbone | input | Maha (rank) | kNN (rank) | MSP |
|---|---|---|---|---|
| ResNet-18 | 224 | 0.956 (1) | 0.936 (2) | 0.574 |
| ResNet-50 | 224 | 0.896 (1) | 0.866 (2) | 0.515 |
| DenseNet-121 | 224 | 0.988 (1) | 0.973 (2) | 0.883 |
| ConvNeXt-Tiny | 224 | 0.928 (1) | 0.916 (2) | 0.846 |
| MobileNetV3-L | 224 | 0.870 (2) | 0.904 (1) | 0.771 |
| RegNetY-3.2GF | 224 | 0.964 (1) | 0.956 (2) | 0.784 |
| EfficientNet-B3 | 300 | 0.885 (2) | 0.934 (1) | 0.803 |
| EfficientNetV2-S | 224 | 0.880 (1) | 0.856 (2) | 0.778 |

EffB3@224 control (appendix): MSP 0.787 (jump survives; not a 300px artifact).

## Methods (first MIDOG mention)

We build on OpenMIBOOD's public MIDOG split and ResNet-50 checkpoint as an external reference. Public R50 ~0.59 is theirs; in-house R50 MSP 0.512 is ours. Never mix.

Related work (locked wording):

> OpenMIBOOD (CVPR 2025) ranks 24 post-hoc detectors on three medical benchmarks with a **single frozen classifier per domain** (ResNet-50 on MIDOG, ResNet-18 on PhaKIR, R(2+1)D on OASIS-3). It shows that ImageNet method ranking does not transfer to medical data and that feature-space scores outperform logit scores under that fixed-architecture protocol. It does not measure whether the detector ranking itself is stable across architectures. We keep their MIDOG split and public ResNet-50 checkpoint as an external reference, and ask the orthogonal question: **holding the dataset fixed, how much does detector ranking move when the backbone changes?**

Full section plan: `manuscript/cvpr2027_outline.md`. Do not draft sections until that outline is approved — especially related work and limitations.
