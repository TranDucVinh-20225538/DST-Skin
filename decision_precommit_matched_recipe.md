# Precommit: matched-recipe ablation — locked 2026-09-18 11:25

Limitations #6: architecture is confounded with its standard optimizer. This is the ablation. Not a new W. Not matched-schedule (cosine 10-epoch is already shared). Not a method.

Do not reinterpret after seeing numbers. Official 8-CNN W stays the **standard-recipe** seed-42 cells.

---

## Fact already in the freeze (do not retrain these)

ResNet-18/50, DenseNet-121, and EffB3 **already share** Adam lr=1e-4 wd=1e-4 bs=64. DenseNet MSP 0.883 and EffB3 MSP 0.803 already jump under that recipe. The load-bearing non-EfficientNet family (DenseNet) is **not** an optimizer confound.

The confound only covers the four families that used a *different* optimizer/wd/bs:

| backbone | official recipe | official MSP |
|---|---|---|
| convnext_tiny | AdamW wd=0.05 bs=64 | 0.846 |
| mobilenet_v3_large | AdamW wd=0.01 bs=64 | 0.771 |
| regnet_y_3_2gf | SGD lr=5e-3 wd=5e-5 | 0.784 |
| efficientnet_v2_s @224 | AdamW wd=1e-5 bs=32 | 0.778 |

If those four still jump under ResNet-Adam, “6/6, 4 families beyond EfficientNet” survives #6. If they do not, that prose dies and A remains on DenseNet (+ EffNet), which the code threshold already allowed.

---

## Two directions

### M1 — jumpers → ResNet Adam (load-bearing)

Four trains, seed 42, frac 1.0, cosine 10 epoch, ImageNet init, hospital-2 OOD.

`Adam lr=1e-4 wd=1e-4 bs=64` on ConvNeXt / MobileNetV3 / RegNet / EffV2-S@224.

Artifacts: `frac1/matched_adam/seed42/` — **cannot** overwrite official `frac1/seed42/`.

Jump vs **frozen** official ResNet mean **0.544654** (threshold MSP ≥ 0.6947). Do not retrain ResNet for this comparison.

`--artifact-tag matched_adam`. Pilot must **not** call `rebuild_architecture_invariance.py`.

### M2 — ResNet → ConvNeXt AdamW (other causal direction)

Two trains: R18 and R50 with `AdamW lr=1e-4 wd=0.05 bs=64`.

Artifacts: `frac1/matched_adamw/seed42/`.

Question: does ResNet **enter the jump band** (MSP ≥ 0.6947)? Not “does DenseNet still beat them.”

---

## Not this plan

Full 8×K recipe grid. SGD-everywhere. ViT retrain. Mixing matched CSVs into official W. Changing `RECIPES` in `cnn_family.py` (seed-variance jobs are in flight on the default recipes). Seed 43–46. Shift-type arms (separate lock).

---

## ID-acc gate (dirty cell)

Camelyon id_val saturates (~0.99). A cell is **dirty** if `best_id_val_acc < 0.97`.

Dirty: report MSP, **do not** count in the 0/4 or 4/4 fraction. If all M1 cells are dirty, the ablation failed (recipe change broke ID training). Stop. Do not interpret “jump died.”

Same class of honesty as Job B (shrink that was R18 getting worse).

---

## Read table (locked before GPU)

`JUMP_DELTA = 0.15`. Frozen ResNet mean = 0.544654.

### M1 (among **clean** cells only)

| result | write |
|---|---|
| ≥3/4 still jump | recipe does not explain 6/6. Keep the 4-family sentence. Limitations #6: “matched Adam on the four mismatched families; jump survived.” |
| 0/4 jump | 6/6 was recipe-inflated. **Drop** “4 families beyond EfficientNet” from §4.1. A still stands via DenseNet (already matched Adam). Do not hide the 0/4 table. |
| 1–2/4 | mixed; list families. §4.1 becomes “DenseNet + the subset that survived matched Adam,” not 6/6. |
| all dirty | ablation uninterpretable. #6 stays as written. Official A unchanged. |

Do not drop a family from the paper to recover 6/6. Do not add a fifth backbone.

### M2

| result | write |
|---|---|
| both R18 and R50 MSP < 0.6947 | ResNet no-jump survives the ConvNeXt recipe. Architecture-side of #6 looks real for ResNet. |
| either MSP ≥ 0.6947 | ResNet **can** jump under AdamW. Cannot write “ResNet vs other families” as a pure architecture split. Qualify: recipe-dependent for ResNet. Official seed-42 A table stays (standard recipes); this is the confound result. |

M1 and M2 can disagree (jumpers still jump under Adam **and** ResNet jumps under AdamW). That is still a finding: both ends move with recipe. Write it. Do not pick the direction that saves the title.

---

## GPU

6 Camelyon trains (4 M1 + 2 M2), then extract+analyze in the same job per backbone. Do not scancel seed-variance 59782–85. Do not overwrite seed 42 official. Wall ~same as one seed-variance backbone (~hours–1 day each).

After CSVs exist: `scripts/camelyon_matched_recipe_read.py`.
