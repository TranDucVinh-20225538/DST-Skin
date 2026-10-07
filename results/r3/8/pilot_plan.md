# R3 item 8 / P2-a: OOD ceiling pilot (plan, fixed before any AUROC is computed)

Purpose: choose the P2-a primary OOD source so that the OOD task is not saturated. The pilot does not
compute any Delta (no same-group vs other-group scorer fits), so it does not touch the P2-a estimand.

Backbone: DINOv2-B (torch.hub facebookresearch/dinov2, x_norm_clstoken), frozen, 224 px
(Resize 224 bicubic + CenterCrop 224 + ImageNet mean/std, images converted to RGB). CPU.

ID sample: NIH ChestX-ray14 `images_001.tar.gz` (first public archive, ~5k images); patient = filename
prefix. Patients split 50/50 with `default_rng(0)` into fit patients and other patients.
- ID_seen: one image held out per fit patient with >= 2 images (up to 300, `default_rng(1)`).
- ID_unseen: 300 images from other patients (`default_rng(2)`).
- Fit set: the remaining fit-patient images (cap 1500, `default_rng(3)`).

OOD candidates, in this fixed order (300 images each, `default_rng(4)` sample of the full list):
1. Kermany pediatric CXR (ChestXRay2017, v2 release, local copy; train + test pooled).
2. Shenzhen Hospital CXR set (NLM, public, adult CXR, 662 images; data.lhncbc.nlm.nih.gov).
Montgomery County (138 images) is excluded: below the P2-a floor of 200 images.

Scorers: mahalanobis_l2, knn_mean_cosine (crossfit_ood, Track A definitions); AUROC with ID positive.

Decision rule (Mahalanobis drives it; kNN reported):
- A candidate saturates if Maha AUROC > 0.97 under ID_seen or ID_unseen.
- Primary = the first candidate in the order above that does not saturate; secondary = the other one.
- If both saturate: primary = the one with the lower max(ID_seen, ID_unseen) Maha AUROC, secondary = the
  other; the P2-a ceiling gate (exclude cells with AUROC > 0.98 under both ID variants) then applies as written,
  and the precommit states that the OOD task may be at ceiling.
- Kermany OCT is dropped from P2-a (different modality; would sit at ceiling by construction).
- One pilot run; no other candidates are added after seeing the result.
