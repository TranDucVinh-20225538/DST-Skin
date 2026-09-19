# Precommit: Viên 1 stain-covariate training — locked 2026-09-18 11:42

Target the Camelyon **hospital/stain covariate**, not compactness. Not a new detector. Not Viên 2. Not a decision to skip CVPR 2027.

Do not overwrite official `frac1/seed42/`. Do not call `rebuild_architecture_invariance.py`. Do not scancel seed-variance / matched-recipe / MIDOG.

---

## Venue (do not lock today)

CVPR 2027 paper **2026-11-16** stays an option. Decide **after Viên 1 numbers** (~early October, ~5 weeks left):

- Miss, or hit but no Viên 2 → write characterization ± this method/kill section, submit CVPR.
- Hit **and** commit to Viên 2 (coverage surrogate, 2–3 months) → then, and only then, discuss skipping CVPR for NeurIPS 2027.

Do not write “we dropped CVPR” in any lock file before that.

---

## What runs

Three backbones, official ResNet-Adam recipe, seed 42, frac 1.0, 10 epoch cosine. **Train-only** HED stain jitter (Ruifrok matrix, Tellez-style, σ=**0.20**, p=**1.0**). Eval / hospital-2 / hospital-1 transforms are the official ImageNet eval — no jitter at test.

| backbone | input | artifact |
|---|---|---|
| resnet18 | 224 | `frac1/stain_cov/seed42/` |
| resnet50 | 224 | same |
| densenet121 | 224 | same |

`--artifact-tag stain_cov --stain-cov --log-msp-epoch`.

Not this viên: ConvNeXt/MobileNet/RegNet/EffNet/ViT, Macenko/staintools, RandStainNA extra dep, color-jitter-only, center-loss, triplet, ensemble, coverage-loss, mixing into official W.

---

## Frozen official reference (hospital 2, seed 42, no stain)

| | MSP | coverage@risk10% (MSP, OOD-triage) |
|---|---|---|
| R18 | 0.574 | 0.145 |
| R50 | 0.515 | 0.149 |
| DenseNet | 0.883 | — |
| Dense−R18 gap | 0.308 | |
| Jump line | MSP ≥ **0.695** (ResNet mean 0.545 + 0.15) | |

---

## Read table (locked before GPU)

`JUMP_DELTA=0.15`. Primary split = hospital **2** (`test`). Held-out = hospital **1** (`val`), never used in train or official A.

A cell is **dirty** if `best_id_val_acc < 0.97`, or DenseNet hospital-2 MSP **< 0.80**. Dirty experiment: report numbers, **do not** call it a gap-shrink. Same class as Job B (R18 getting worse).

R18 (resp. R50) **hits the bar** on a split iff not dirty and:

- MSP ≥ 0.695, **or**
- (DenseNet − this ResNet) ≤ 0.154 **and** this ResNet MSP ≥ its official floor (R18 **0.574**, R50 **0.515**).

Phao B: R18 hospital-2 MSP coverage@risk10% ≥ **0.135** (official 0.145 minus 0.01 slack). Below that: method may lift AUROC while hurting triage — **not** a clean hit.

| result | write |
|---|---|
| Dirty | Job-B-class fail. Appendix kill. No Viên 2. CVPR still on the table. |
| Hospital-2 R18 misses bar | Viên 1 miss. One more method viên allowed later, then stop. Characterization ± kill section. Decide CVPR after this miss. |
| Hospital-2 R18 hits, R50 does not, coverage ok, hospital-1 R18 misses or hits | **Weak hit.** Method paragraph on 3 anchors. Do **not** expand to 8. Do **not** open Viên 2. CVPR still on the table. |
| Hospital-2 R18 **and** R50 hit, coverage ok, hospital-1 R18 **also** hits | **Strong hit (3/3 + held-out).** Only this branch: consider 8-CNN stain-cov **and** whether to open Viên 2 — that is the CVPR-vs-NeurIPS discussion, not today. |

Do not drop DenseNet to make the gap look small. Do not retune σ after seeing AUROC. Do not mix stain CSVs into official W.

Hospital-1 numbers are reported even on a miss. They do not rewrite official A.

---

## After numbers (not now)

Write one subsection. If miss: appendix. If weak/strong hit: method section, still secondary to cửa 2 + Phao B unless strong hit + Viên 2 is chosen later.
