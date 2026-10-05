# P1 — ResNet50 slide-disjoint AUROC < 0.5 (patch 1, seed 42)

Precommit: `decisions/precommit_isbi_patch2_2026-10-06.md`, P1. Checks in the precommitted order.

## (a) Score sign and ID/OOD orientation

- fold 0: independent recompute (own softmax/logsumexp, sklearn, ID label 1, higher = ID, float32 scores as in `calc_auroc`) MSP 0.3163 / Energy 0.3078 vs reported 0.3163 / 0.3078 -> match. In float64 the MSP AUROC is 0.3164: 1.1% of MSP values round to exactly 1.0 in float32 (ties); |diff| <= 0.001, same convention as every published AUROC, not a sign or alignment issue.
- fold 1: independent recompute (own softmax/logsumexp, sklearn, ID label 1, higher = ID, float32 scores as in `calc_auroc`) MSP 0.2273 / Energy 0.2260 vs reported 0.2273 / 0.2260 -> match. In float64 the MSP AUROC is 0.2264: 18.9% of MSP values round to exactly 1.0 in float32 (ties); |diff| <= 0.001, same convention as every published AUROC, not a sign or alignment issue.
- The same scoring code gives ConvNeXt / DenseNet slide-disjoint AUROC 0.47-0.65 and seen-slide 0.73-0.79 (patch-1 table), so a global sign flip is excluded; the published ResNet50 seed-42 MSP is 0.513 (all id_val) with the same convention.
- Row alignment: seen-slide id_val accuracy 0.995-0.997 (id_val order) and ConvNeXt OOD accuracy 0.82-0.85 through the identical OOD loader (OOD order); published labels equal metadata order.

## (b) Per-fold accuracy and AUROC, seen vs unseen slides

| fold | acc ID seen | acc ID unseen | acc OOD | pred. tumour OOD | MSP unseen | MSP seen | Energy unseen | Energy seen |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.997 | 0.617 | 0.506 | 0.009 | 0.316 | 0.556 | 0.308 | 0.546 |
| 1 | 0.995 | 0.923 | 0.505 | 0.005 | 0.226 | 0.472 | 0.226 | 0.472 |

Published seed-42 ResNet50 (all train slides): OOD accuracy 0.550, predicted tumour on 0.060 of OOD (true 0.500).

## (c) Training collapse

- fold 0: loss 0.0487 0.0292 0.0239 0.0199 0.0165 0.0120 0.0088 0.0063 0.0038 0.0030; train tumour fraction 0.255 (n = 97099).
- fold 1: loss 0.0533 0.0383 0.0342 0.0300 0.0257 0.0218 0.0177 0.0143 0.0114 0.0097; train tumour fraction 0.617 (n = 205337).
- The patch-1 folds differ in class balance as well as size (tumour fraction 0.255 vs 0.617), a further confound of the patch-1 fold comparison; the v2 JSONs record the balanced folds' tumour fraction.
- Predictions are not constant on ID: seen-slide accuracy 0.995-0.997, both classes predicted. On hospital 2 the classifier predicts tumour on <1% of patches (table above) with saturated confidence; this is a collapse under the hospital shift, not a training failure.

## (d) Checkpoint selection (best vs last epoch)

- fold 0: best epoch 10 (sel_acc 0.9966), last epoch 10 (sel_acc 0.9966) -> best = last.
- fold 1: best epoch 10 (sel_acc 0.9950), last epoch 10 (sel_acc 0.9950) -> best = last.
- The v2 runs save both checkpoints and report the last-epoch AUROC as a diagnostic.

## Verdict

No technical bug (sign, orientation, alignment, subset, checkpoint). The patch-1 ResNet50 numbers (MSP 0.316 / 0.226 slide-disjoint) are kept and labelled **anomalous fold**: classifier collapse to 'normal' on the OOD hospital makes OOD patches more confident than unseen ID slides. The within-model seen-minus-unseen gap (0.24-0.25) is unaffected by this reading. Hyperparameters unchanged.

