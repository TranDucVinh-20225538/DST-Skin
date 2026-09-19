# Precommit: Camelyon shift-type 2-arm extract — locked 2026-09-18 11:20

Raise the claim from “ranking moves with backbone on this split” to **whether the Camelyon MSP-jump is shift-conditional**. Not a new detector. Not a fourth medical domain. Matched-recipe is the other lock, not this file.

Do not reinterpret after seeing numbers. Official 8-CNN W stays **hospital-2, seed 42**. These arms never enter Kendall W.

---

## Question

Same 8 Camelyon classifiers, same ID (Camelyon `id_val`), two new OOD sets. Jump formula unchanged:

\[
\mathrm{jump}_X = \mathrm{MSP}_X - \tfrac12\bigl(\mathrm{MSP}_{\mathrm{R18}}+\mathrm{MSP}_{\mathrm{R50}}\bigr)
\]

Hit iff \(\mathrm{jump}_X \ge 0.15\) (MSP ≥ 0.6947 on the frozen covariate arm; on new arms the ResNet mean is that arm’s own R18/R50 MSP).

We already have the covariate cell. The missing cells are **task-shift H&E** and **far-OOD natural**. OpenOOD v1.5 Fig. 3 is semantic natural; iWildCam / skin / MIDOG zoo change dataset and shift together. This is the only execution-risk experiment that can answer “v1.5 already” without training.

**2511.11934 already ranks confidence scores by shift severity on CIFAR.** Do not write that we discovered “ranking depends on shift type.” The increment is: the **structured Camelyon MSP-jump** (ResNet vs the other six) is or is not shift-conditional on this zoo.

---

## Arms

| arm | ID | OOD | role |
|---|---|---|---|
| **C** covariate | Camelyon `id_val` | hospital 2 | **frozen**. Do not re-extract. Numbers from official seed-42 CSVs. |
| **T** task-shift H&E | Camelyon `id_val` | MIDOG **1a only** (train+id_val+id_test imglists, unlabeled) | mitosis vs lymph-node tumor; still H&E. Do **not** include 1b+1c (that is MIDOG’s own covariate). |
| **F** far-OOD | Camelyon `id_val` | CIFAR-10 **test** | OpenOOD-style semantic natural. |

Do not call arm T “semantic OOD” in the OpenOOD near/far sense. Call it task-shift H&E. Arm F is far-OOD / semantic natural.

**Not this plan:** SVHN, CIFAR-10-C, Texture, skin-as-OOD, NCT-CRC, a new loss, extra backbones, mixing seed 43–46, overwriting `*_features.pt`. Matched-recipe is a **separate** lock (`decision_precommit_matched_recipe.md`), not this extract.

---

## What runs (extract only)

Official 8, seed 42, `frac1`, existing checkpoints. **No train.**

| backbone | input | checkpoint stem |
|---|---|---|
| resnet18 | 224 | `resnet18` |
| resnet50 | 224 | `resnet50` |
| densenet121 | 224 | `densenet121` |
| convnext_tiny | 224 | `convnext_tiny` |
| mobilenet_v3_large | 224 | `mobilenet_v3_large` |
| regnet_y_3_2gf | 224 | `regnet_y_3_2gf` |
| effb3 | **300** | `effb3` |
| efficientnet_v2_s | **224** | `efficientnet_v2_s_224` |

ViT-B/16: extract as a held-out row, **not** in the 6/6 count.

Reuse `train_*` / `val_*` / `fc_*` from existing `outputs/features/camelyon17/frac1/seed42/{stem}_features.pt` (Maha/kNN fit on Camelyon train; ID scores on Camelyon id_val). Extract **only** the new OOD tensors.

Transform: **Camelyon eval / ImageNet mean-std**, including on MIDOG crops. Do not use MIDOG adapter mean/std — those classifiers never saw it.

Artifacts (new paths only):

- `outputs/features/camelyon17/frac1/seed42/shift_type/{stem}_{arm}_ood.pt`
- `outputs/reports/camelyon17/frac1/seed42/shift_type/{stem}_{arm}_score_comparison.csv`
- `outputs/reports/camelyon_shift_type.txt` (read-table verdict)

Do not write into official `*_features.pt` or official `*_score_comparison.csv`. Do not call `rebuild_architecture_invariance.py`.

GPU: one job, sequential 8×2 extracts. Do **not** scancel seed-variance (59782–59785) or MIDOG zoo (59776). Submit when a GPU is free (59776 finishing is the natural slot). Wall-clock: extract-only, hours not a day.

---

## Read table (locked before GPU)

`JUMP_DELTA = 0.15`. Apply **separately** to arm T and arm F. Arm C is the frozen reference (6/6 non-ResNet jump).

### 1. Saturation kill (per arm)

An arm is **uninformative** if median MSP across the 8 CNNs ≥ **0.95**, or min MSP ≥ **0.90**.

Then: drop that arm. Jump ≈ 0 is the ceiling, **not** “no architecture effect.” Write `saturated; arm dropped.` Do not use a saturated arm to claim shift-conditionality.

If **both** T and F saturate: the experiment failed to raise the ceiling. Stop. Do not add a fourth OOD to rescue it.

### 2. If the arm is informative

Count how many of the **6 non-ResNet** official CNNs jump vs **that arm’s** ResNet mean.

| result | write |
|---|---|
| **6/6** jump | jump is **not** covariate-specific on this zoo. Do **not** claim a shift-conditional law. Increment vs OpenOOD v1.5 shrinks; paper stays characterization + Phao B. |
| **1–5 / 6** jump | mixed; list which families; **not** a law. Report, do not hide, do not drop a backbone to get 0/6 or 6/6. |
| **0/6** jump (and the 6-family mean jump < 0.15) | **HIT**: hospital-2 jumps, this arm does not. That is the ceiling raise. One results paragraph + a 2×3 MSP table (C / T / F × backbone). |

R18/R50 themselves jumping on a new arm is variance/shift, not a new A. Report it; do not rewrite Camelyon A.

### 3. Joint readout (only after both arms are classified)

| T | F | paper sentence |
|---|---|---|
| 0/6 | 0/6 or saturated | MSP-jump is hospital-shift specific on this zoo. v1.5 semantic natural is a different phenomenon. |
| 6/6 | 6/6 | classifiers, not shift type. Do not sell “covariate vs semantic.” |
| 0/6 | 6/6 | odd; report as far-OOD vs H&E-task split, not a slogan. |
| 6/6 | 0/6 or saturated | jump survives other H&E tasks; far-OOD uninformative or opposite. Not the v1.5 counter we wanted; still report. |
| mixed on either | — | no law. Table in appendix. Official A unchanged. |

Do not pool Kendall W across arms. Optional per-arm W is appendix only, not a new official number.

---

## After numbers (not now)

If HIT (T = 0/6, C = 6/6): add one §4.x paragraph and one limitations sentence that 2511.11934 already varies shift severity on natural images. Do **not** rewrite contributions bullet 1 into a discovery of architecture-sensitive ranking.

If miss (T or F = 6/6 informative): do **not** bury the table. Calibration stays ~40–50%. Official A and Phao B unchanged.

If seed-variance later hits the middle branch, that clause and this shift-type paragraph are independent.

---

## Order

1. Lock this file (done).
2. Script extract + read-table (`scripts/camelyon_shift_type_ood.py`, `scripts/camelyon_shift_type_read.py`).
3. GPU when a card is free. Not before.
4. Apply this read table. Then, and only then, one outline paragraph.

Writing related work stays in parallel and does not wait for these arms.
