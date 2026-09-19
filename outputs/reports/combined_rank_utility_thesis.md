# Combined thesis: rank instability (i) + utility mismatch (ii)

Locked 2026-09-17. CPU-only draft from existing CSVs. No new GPU.
Does not wait for zoo `59493` or remaining FM cells. Kendall's W is **per domain**, never pooled.

Addendum 2026-09-18: this combined thesis **is** the paper. Maha>MSP is not a fallback if (i) fails — OpenMIBOOD §5 already published feature>logit on medical. If Camelyon-8 is C, rescue with Direction #1/#3; do not rewrite this file into a Maha>MSP note.

Phao A is **not** in this draft as a headline. Scope: FM amplifies a Maha–MSP gap that already exists on CNN; upgrade only if GigaPath-Camelyon or Phikon-Camelyon invert (MSP AUROC < 0.5, sign-check `AUROC + AUROC_flipped = 1`). Until then: MIDOG-specific footnote (`mismatch=severe`).

## Core claim

OOD evaluation is wrong in two independent ways at once:

1. **Rank instability (cửa 2).** Which method looks best depends on the backbone. Founding Camelyon cell: MSP AUROC 0.51–0.57 on ResNet-18/50 vs **0.803** on EffB3@300, and **0.787** on EffB3@224 (resolution control — not a 300px artifact).
2. **Utility mismatch is domain-dependent unpredictability, not a universal reversal.** AUROC rank and OOD-triage coverage@risk10% rank **disagree on both domains**, but **the direction of the disagreement flips**. Do not claim “AUROC-winner = utility-loser” as a law — that pairing is Camelyon-specific (n=1 domain) and is exactly the single-domain overclaim that desk-rejected the old CMPB triage paradox.

Actionable warning: do not pick backbone/method from an AUROC leaderboard for triage; check coverage@risk on the target domain. The sign of the AUROC↔coverage mismatch is **not predictable** from Camelyon alone.

## Trục (i) — Kendall's W (not pooled)

Locked n=3, 7 methods (MSP, Energy, ELogitNorm, ViM, ReAct, Maha, kNN), from `architecture_invariance_kendall_w.csv`:

| domain | n_backbones | Kendall's W |
|---|---|---|
| Camelyon17 | 3 (R18, R50, EffB3@300) | 0.714 |
| MIDOG | 3 (R18, R50 in-house, EffB3@300) | 0.825 |
| Skin ISIC→PAD | 3 | 0.873 |

W=1 identical method ranking across backbones; W=0 none. Camelyon W=0.714 is the cửa-2 number. Do not average with MIDOG/skin.

Same four common methods (MSP, Energy, Maha, kNN) after adding EffB3@224:

| domain | backbones | Kendall's W |
|---|---|---|
| Camelyon17 | R18, R50, EffB3@300, EffB3@224 | 0.850 |
| MIDOG | R18, R50 in-house, EffB3@300 | 0.911 |

On Camelyon the top-1 method among those four is **Mahalanobis** on both ResNets and EffB3@224, but **kNN** on EffB3@300. That is the rank flip. MSP AUROC still jumps on both EffB3 resolutions.

MIDOG R50 uses **in-house** `midog/seed42/resnet50_score_comparison.csv` (MSP 0.512), not OpenMIBOOD 0.59.

## Trục (ii) — OOD-triage coverage@risk 10% (MSP)

Protocol: rank OOD samples by MSP (higher = more ID-like), measure classifier error on the accepted prefix, take the largest coverage whose risk ≤ 10%. Same script for both domains (`pathology_risk_coverage.py`). ID-selective Camelyon is **vacuous** (ID-val acc ≈ 99.5%, coverage@risk10% = 1.0 for every backbone). Do not cite it.

### Camelyon (hospital-2)

| backbone | family | MSP AUROC | MSP rank among {MSP,Energy,Maha,kNN} (1=best) | top-1 | coverage@risk10% | AURC (lower better) |
|---|---|---|---|---|---|---|
| ResNet-18@224 | resnet_like | 0.574 | 3 | Mahalanobis | 0.145 | 0.342 |
| ResNet-50@224 | resnet_like | 0.515 | 3 | Mahalanobis | 0.149 | 0.346 |
| EffB3@300 | efficientnet_like | 0.803 | 3 | kNN | 0.107 | 0.295 |
| EffB3@224 | efficientnet_like | 0.787 | 4 | Mahalanobis | 0.013 | 0.256 |

AUROC order: **EffB3@300 > EffB3@224 > ResNet-18@224 > ResNet-50@224**
coverage@risk10% order: **ResNet-50@224 > ResNet-18@224 > EffB3@300 > EffB3@224**

On Camelyon, AUROC winner **EffB3@300** (MSP 0.803) is not the coverage@risk10% winner (**ResNet-50@224**, 0.149). Coverage loser: **EffB3@224** (0.013). This is an existence proof on one domain, not a law.

### MIDOG (in-house R18 / R50 / EffB3@300) — direction does not copy Camelyon

Protocol identical. n_OOD=5902. ID n_val=251 is small; MSP AUROC spread is tiny (0.438–0.512). Do not over-read “winner” labels here. Full table, not a single EffB3 cell:

| backbone | family | MSP AUROC | coverage@risk10% |
|---|---|---|---|
| ResNet-18@224 | resnet_like | 0.438 | 0.442 |
| ResNet-50@224 | resnet_like | 0.512 | 0.166 |
| EffB3@300 | efficientnet_like | 0.486 | 0.712 |

AUROC order: **ResNet-50@224 > EffB3@300 > ResNet-18@224**
coverage@risk10% order: **EffB3@300 > ResNet-18@224 > ResNet-50@224**

AUROC-highest: **ResNet-50@224** (0.512).
coverage@risk10% highest: **EffB3@300** (0.712).
coverage@risk10% lowest: **ResNet-50@224** (0.166).

EffB3 does **not** win both on MIDOG. It wins coverage (0.712) but not AUROC (0.486 vs R50 0.512). R50 is the AUROC-highest backbone and the coverage-lowest (0.166). That is still not a license to write “AUROC-winner = utility-loser” as a universal rule: MIDOG AUROC gaps are small, and the **permutation of ranks is not the Camelyon permutation**.

**Claim that survives both tables:** the AUROC↔coverage mapping is **unreliable**, and **the direction of unreliability is domain-dependent**. Camelyon: EffNet tops AUROC and sits at the bottom of coverage. MIDOG: the ranking permutes the other way (R50 tops AUROC / bottoms coverage; EffB3 bottoms-to-mid AUROC / tops coverage). This is instability stacked on cửa 2, not a second copy of the Camelyon reversal. Reviewer-safe sentence: *do not choose a backbone for triage from an AUROC leaderboard; measure coverage@risk on the deployment domain, because the sign of the mismatch cannot be predicted from Camelyon.*

Do not pool Kendall's W with these coverage ranks.

## What this is not

- Not “EffB3 always loses triage.” False on MIDOG.
- Not a frozen-FM headline (Phao A). UNI-Camelyon MSP is 0.54–0.60, not invert. CNN R18-MIDOG MSP is already 0.438.
- Not zoo A/B/C. n=4 Camelyon spans 2 families; A/B/C still waits for n=8 (`59493`). After zoo, test whether **unpredictability** (rank disagreement) spans families — not whether EffB3-as-utility-loser does.
- Not ID-val risk-coverage. That protocol saturates on Camelyon and on skin actually favored EffB3. The mismatch is **OOD-triage**.

## Next (no GPU now)

2. Zoo `59493` done → rerun OOD-triage coverage@risk10% on all 8 Camelyon backbones (add DenseNet121, ConvNeXt-Tiny, MobileNetV3-L, RegNetY-3.2GF, EffV2-S@384). Question: does AUROC↔coverage **disagreement** span families, not “does EffB3 always lose.”
3. GigaPath-Camelyon + Phikon done → rerun `scripts/audit_fm_msp_sign.py`. Invert on Camelyon → upgrade Phao A. Else keep MIDOG-only footnote.
