# Precommit: #5 iWildCam + #4 ViT-B/16 — locked 2026-09-18 02:46

Execution risk, not design risk. Hard stop **2026-10-02** (2 weeks). After that: numbers go in the paper if they exist; otherwise one future-work sentence. Do not eat writing time. Do not steal GPU from Job B (`59688`) or Camelyon EffV2-S@224 (`59687`).

#5 is the heavier CVPR lever. #4 is cheaper epistemology. Both may run in parallel once GPUs are free.

---

## #5 — iWildCam (WILDS) — heaviest remaining CVPR lever

**Why:** OpenMIBOOD = medical + 1 backbone. A medical-only 8-backbone paper can still be read as “OpenMIBOOD expanded.” If detector ranking still moves with architecture on a **non-medical** covariate shift, the finding is about OOD detection under shift, not a medical benchmark.

**Dataset:** iWildCam (WILDS). Reuse the Camelyon WILDS loader pattern. Shift = camera-trap location, analogous to hospital. FMoW is backup only if iWildCam data is blocked; do not start FMoW in parallel.

**Pilot (not full 8 in two weeks):** ResNet-18, ResNet-50, DenseNet-121. Same 7 scores, same `JUMP_DELTA=0.15` vs **that domain’s** ResNet MSP mean. Do not pool Kendall W with Camelyon/skin.

**Read the result; do not reinterpret:**

| iWildCam | what it is |
|---|---|
| MSP jump ≥0.15 on DenseNet vs ResNet mean | architecture-instability is **not** medical-only. Headline can leave “medical benchmark.” |
| No jump (or only EfficientNet, which is not in this pilot) | **bound**, not a failed experiment: phenomenon may be biomedical-specific. Paper stays Camelyon/skin. Venue shape slides toward workshop/CMPB. Do not hunt a fourth medical domain to “save” #5. |

Do not add MIDOG-8 to rescue a negative iWildCam.

---

## #4 — 9th backbone = ViT-B/16 @224 on Camelyon (not another CNN)

**Why:** n=8 is a description. A locked guess before eval is a prediction. ViT also tests whether the pattern leaves the CNN family (ConvNeXt already covers “modern CNN ≈ Swin”). **Do not use Swin-T as the 9th** unless ViT cannot be trained; Swin-T is too close to ConvNeXt-Tiny.

**Regression on depth / params / RF was fit on the 8 Camelyon CNNs before any ViT train. It does not work. Do not quote it as the predictor.**

```
in-sample R² = 0.17
LOO R²      = -5.15
LOO MAE     = 0.226  (jump threshold is 0.15 — model is worse than the threshold)
```

LOO misses both ResNets (predicts jump) and DenseNet (predicts no jump). Linear log(params), log(depth), log(RF) **cannot** see the actual split (resnet_like vs not). Point estimates ViT-B/16 +0.25 / Swin-T +0.27 would say “jump” but that is an overfit intercept, not a validated model.

**Locked prediction (family prior, the only rule that fits n=8):**

- ResNet mean MSP = **0.544654** (R18 0.5743, R50 0.5150). Jump iff MSP − mean ≥ **0.15** ⇔ MSP ≥ **0.6947**. Same A/B/C constants.
- On n=8, non-`resnet_like` backbones all jumped (6/6). ViT-B/16 is not `resnet_like`.
- **Precommit: ViT-B/16 Camelyon MSP jumps (MSP ≥ 0.6947).**
- Secondary, not scored: Maha/kNN stay rank 1–2 (cửa-2 feature stability). If Maha is dragged >0.10 vs median(R18,R50,EffB3), that is scenario C for this cell only — do not rewrite Camelyon-8 A.

**Scoring after eval (do not move the line):**

| outcome | write |
|---|---|
| MSP ≥ 0.6947 | prediction hit; pattern is not CNN-only |
| MSP < 0.6947 | prediction miss; MSP-jump may be CNN-family. Still a result. Do not add another CNN to “get a hit.” |

Fit of a new regression **after** seeing ViT is forbidden. The family-prior call above is the only precommit.

Skin ViT is optional and must not delay Camelyon ViT.

---

## Order

1. Do not **steal** GPU from `59688` / `59687`. Queueing behind them is allowed.
2. CPU first: iWildCam download (`dst-iwild-dl`). After `metadata.csv` exists, queue `dst-iwild` GPU (R18+R50+DenseNet). Parallel GPU slot: `dst-vit-cam` Camelyon ViT-B/16.
3. 2026-10-02: stop. Partial tables are OK. No extra backbones on iWildCam beyond the three unless both #4 and the iWildCam-3 already landed before the date.

---

## Scored 2026-09-18 — do not reinterpret

**#4 HIT.** ViT-B/16 Camelyon MSP = **0.884** ≥ 0.6947. Maha 0.958 / kNN 0.941, rank 1–2. Not mixed into official 8-CNN W. Write as a locked prediction on a held-out family, not as a 9th W column.

**#5 MISS = bound.** iWildCam DenseNet MSP 0.575 vs ResNet mean 0.604, jump **−0.029**. Architecture-instability of MSP may be biomedical-specific. One limitations sentence. Do not hunt a fourth domain. Do not start FMoW. Do not add MIDOG-8.
