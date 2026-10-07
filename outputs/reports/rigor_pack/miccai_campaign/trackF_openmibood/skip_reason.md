# Track F (OpenMIBOOD audit): skipped — not eligible

Precommit L1: run only if per-image case IDs are recoverable for >= 90% of the images of the existing MIDOG
caches, using those caches only.

Checked (no scores computed):
- The OpenMIBOOD MIDOG imglists carry the case in the path (`1a/<case>/<case>_<tile>_<label>.tiff`): 40 training
  cases; the ID test (`test_midog.txt`, cases 003 / 017 / 025 / 027 / 049) and ID validation lists are
  case-disjoint from training by construction, so the benchmark has no group-shared ID set.
- The existing MIDOG feature caches (`outputs/features/midog/seed*/<arch>_features.pt`) store the training
  split in shuffled loader order (`midog_ood.get_dataloaders` uses `shuffle=True` for train): the cached
  training labels agree with the imglist order for only 38.7% of the rows (convnext_tiny seed 42), so case IDs
  of the cached training images cannot be recovered. The ID-test cache is in imglist order but has no
  group-shared counterpart.
- A group-shared vs group-clean audit would need a new GPU re-extraction of the training split; the precommit
  restricts Track F to the existing caches. Not run. Never part of the L7 denominator.
