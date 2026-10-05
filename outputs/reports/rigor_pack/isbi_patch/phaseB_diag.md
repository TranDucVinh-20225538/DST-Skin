# Phase B per-fold table and diagnostics

Seed 42, 3 backbones, 2 H11c slide folds; a sensitivity check, not a general claim.
Primary reading here: the within-model gap (same retrained model, id_val on seen slides minus id_val on unseen slides, same OOD set), because fold 0 trains on fewer slides and is weaker on unseen slides (published-vs-retrain delta mixes slide leakage with a weaker model).

| arch | fold | acc ID seen | acc ID unseen | acc OOD | MSP unseen | MSP seen | gap MSP | Energy unseen | Energy seen | gap Energy | published MSP (same ID subset) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 0 | 0.997 | 0.617 | 0.506 | 0.316 | 0.556 | 0.240 | 0.308 | 0.546 | 0.238 | 0.527 |
| resnet50 | 1 | 0.995 | 0.923 | 0.505 | 0.226 | 0.472 | 0.245 | 0.226 | 0.472 | 0.246 | 0.486 |
| convnext_tiny | 0 | 0.997 | 0.667 | 0.817 | 0.470 | 0.730 | 0.259 | 0.526 | 0.807 | 0.280 | 0.855 |
| convnext_tiny | 1 | 0.997 | 0.975 | 0.853 | 0.653 | 0.794 | 0.141 | 0.691 | 0.821 | 0.130 | 0.827 |
| densenet121 | 0 | 0.997 | 0.617 | 0.657 | 0.573 | 0.789 | 0.216 | 0.570 | 0.784 | 0.214 | 0.887 |
| densenet121 | 1 | 0.996 | 0.959 | 0.665 | 0.644 | 0.793 | 0.149 | 0.640 | 0.783 | 0.143 | 0.874 |

Within-model gap range over 6 (arch, fold) cells: MSP 0.141-0.259, Energy 0.130-0.280.

## ResNet50 below chance

- Not a sign flip: MSP/Energy use the same code path (higher = ID) as ConvNeXt/DenseNet, which stay above 0.5; the published ResNet50 is already at chance (MSP 0.507 on the same ID subsets).
- Not a failed training run: loss falls monotonically to <0.006, checkpoint-selection accuracy 0.995-0.997, accuracy on seen id_val slides 0.995-0.997.
- Failure mode: on hospital 2 the ResNet50 recipe collapses to the 'normal' class (predicts tumour on 0.5-0.9% of OOD patches vs 50% true tumour; OOD accuracy 0.505-0.506; published model 0.550) with near-saturated confidence, so OOD patches score as more confident than unseen ID slides. MSP/Energy AUROC < 0.5 is therefore a confident-collapse case of the classifier, not evidence about leakage by itself; the within-model gap (seen minus unseen slides) is still 0.240-0.245.
- The sign-flipped MSP AUROC (column auroc_msp_unseen_flipped_diag) is a diagnostic only, chosen after seeing labels, and is not reported as a detector.
- ResNet50 is kept in the table (precommit: no dropping archs after seeing deltas). The precommitted Phase B bar (|delta| > 0.02 on >= 2/3 archs) is met by ConvNeXt and DenseNet alone.

