# Checklist for post-hoc OOD benchmarks with grouped data

1. Use a group-disjoint ID split (no slide / patient / location / experiment shared between the scorer's fit set and ID eval), or the cross-fitted scorer (F2) if the full ID set must be kept.
2. Ledoit-Wolf covariance in float64 for Mahalanobis.
3. Fix the BLAS thread count (OMP / OPENBLAS / MKL) and report it; results move at the 1e-6 level otherwise.
4. Report >= 1 seed × >= 6 archs (sample_size_disjoint: min reliable disjoint partition seed 1, arch 6; the Maha/ViM 8-score tree did not stabilise within 6 archs).
5. No candidate statistic (M2, M3, centroid distance) predicts Δ across datasets; report the group-disjoint AUROC itself.
