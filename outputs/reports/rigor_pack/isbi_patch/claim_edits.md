# Suggested claim edits (not applied; manuscript untouched)

Based on `README.md` in this folder and `decisions/precommit_isbi_patch_2026-10-05.md`.

1. Feature-space default (H5a). Replace any "feature-space scores are best on every backbone and
   seed" with: "feature-space wins on 5/8 backbones under slide-disjoint scoring (seed 42)".
   With the original same-slide id_val it is 8/8; with leak-free Maha/kNN only, 4/8.
2. ID-side slide sharing. Mahalanobis, kNN (H11c) and ViM lose 0.04-0.21 AUROC on 8/8 backbones
   when the ID patches come from slides not used to fit the scorer; report the slide-disjoint
   AUROCs in the appendix. ReAct changes by <= 0.005 (one sentence).
3. Logit vs feature comparison (sensitivity check: 3 backbones, seed 42, 2 slide folds; not a
   general claim). Primary number: the within-model gap, i.e. the same retrained model scored with
   id_val from seen vs unseen slides against the same OOD set: MSP 0.14-0.26, Energy 0.13-0.28 over
   all 6 (arch, fold) cells (per-fold table: `phaseB_per_fold.csv`, `phaseB_diag.md`). The
   published-vs-retrain delta (0.24-0.28) is secondary because fold 0 trains on 1/3 of the slides
   and reaches only 0.62-0.67 accuracy on unseen slides (fold 1: 0.92-0.98), so it mixes leakage
   with a weaker model. ResNet50 < 0.5 is not a sign flip or failed run: the recipe collapses to
   'normal' on hospital 2 (tumour predicted on <1% of OOD patches, OOD accuracy 0.51) with saturated
   confidence; report it flagged, not as leakage evidence on its own. The precommitted bar holds on
   ConvNeXt and DenseNet alone.
4. How many architectures. Drop the earlier ">= 4 architectures" recommendation. New wording: "up to
   6 architectures are still not enough for a reliable detector ranking on this setup" (disjoint
   partitions: median tau 0.52-0.62 at every m in 2..6; top-1 still >= 0.96). Seeds: 1 seed meets
   the bar; OOD patients: 7 of 9 under disjoint partitions (2 under overlapping subsets).
5. H15. Cite as robust only ConvNeXt (5/5) and MobileNetV3 (4/5); RegNetY seed-dependent (3/5);
   EffV2-S, ResNet18-AdamW, ResNet50-AdamW did not replicate the seed-42 jump.
6. Scope. All of the above is Camelyon17 (seed 42 for leakage) plus the existing MIDOG H12
   one-liner; no claim beyond these setups.

Proposed claim:
"Under slide-disjoint evaluation on Camelyon17 (seed 42), every train-fitted score except ReAct loses
more than 0.02 AUROC, and feature-space scores lead on only 5/8 backbones. In a 3-backbone retraining
check, MSP and Energy also drop on unseen slides, so current logit-versus-feature comparisons are not
protocol-matched."
