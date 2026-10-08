# R3 item 8 / P2-d: OOD ceiling pilot (plan, fixed before any AUROC is computed)

Follows results/r3/8/p2d/PRECOMMIT.json ("pilot", candidate lists, "brain_rules", "stopping_rule"). The pilot
computes OOD AUROC only; no same-group vs other-group scorer fits, so no Delta.

Backbone: DINOv2-B (torch.hub facebookresearch/dinov2, x_norm_clstoken), frozen; staged 256 px PNGs
(data/staged/p2d_*_256), Resize 224 bicubic + CenterCrop 224 + ImageNet mean/std, converted to RGB. CPU.

Sets per candidate, from the Phase 0 tables (commit 73683e9):
- Kvasir-Capsule: `kvasir_sets.csv.gz`, K_seg roles (`role_seg`). Brain: `brain_sets.csv.gz`, record-wise roles (`role`).
- Fit: 1,500 images sampled from role `fit` (`default_rng(3)`); all of them if fewer (brain: 1,044 / 958 / 750).
- ID_seen: 300 images sampled (`default_rng(1)`) from role `seen` whose video / patient has >= 1 image in the
  sampled fit set.
- ID_unseen: 300 images sampled (`default_rng(2)`) from role `unseen` (held-out videos / patients).
- OOD: 300 images sampled (`default_rng(4)`) from role `ood`.
Sizes below 300 use all eligible images; actual counts are written to the output.

Candidates, in the precommitted order:
- Kvasir-Capsule: 1. Angiectasia, 2. Erosion, 3. Galar PillCam SB3 small-intestine frames, videos 61-70.
- Brain MRI: 1. meningioma, 2. pituitary, 3. glioma.
Kvasir 1-2 and brain 1-3 run in one job. Galar runs only if Angiectasia and Erosion are both skipped. The
PRECOMMIT does not fix Galar's ID sets; if Galar is reached, an addendum fixing them (and the frame sample) is
committed before the Galar pilot runs.

Scorers: mahalanobis_l2, knn_mean_cosine (crossfit_ood, Track A definitions); AUROC with ID positive.

Decision rule (Mahalanobis drives it; kNN reported), per dataset:
- A candidate is skipped if Maha AUROC > 0.97 under ID_seen or ID_unseen.
- Primary = first non-skipped candidate in the order above; secondary = the next non-skipped one.
- Brain: if all three classes are skipped, brain P2-d is "ceiling - not identifiable" and stops (PRECOMMIT).
- Kvasir-Capsule: if all three candidates are skipped, primary = the one with the lowest max(ID_seen, ID_unseen)
  Maha AUROC; the ceiling gate (AUROC > 0.98 under both) then applies as written and the report states that the
  OOD task may be at ceiling (same rule as the P2-a pilot).
- The pilot's seen - unseen difference is disclosed, not interpreted (small sample, frozen backbone only).
- One pilot run per dataset; no candidates added after seeing results.
