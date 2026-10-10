# Item 0 — BUDGET (planning estimates re-derived from the item 0 microbenchmarks; still estimates, not measurements)

Measured (A100-SXM4-80GB, float64): matmul 17.8 TFLOP/s; `inv` d = 2560 0.015 s; `cholesky` d = 2560 0.002 s;
brute-force kNN 50k × 100k × 2560 with top-200 1.9 s; full Camelyon 2-fold kNN Δ (resnet50, d = 2048) 14–19 s on GPU vs 160–1,290 s on CPU.

| Item | Dominant cost | Re-derived planning estimate (GPU-h) | Hard cap (unchanged) |
|---|---|---|---|
| A | LOGO / K-fold refits (≤ 15 groups → ≤ 15 covariance downdates per fold, d ≤ 2560) on 27 cells; A2 13 backbones × 2 folds × 30 designs × 10 repeats = 7,800 LW fits + scoring of ≤ 120k points each | 3–6 | 12 |
| B | one top-200 search per (fit, eval set): ~27 cells × 2 folds × ~120k queries vs ≤ 205k fit points ≈ 20 s each, plus MTS toy (16 replicates, d ≤ 1024, bisection) | 2–6 | 18 |
| C | 6,000 C1 configs × adaptive 8–64 replicates; per replicate a d×d Cholesky + N·d² products + 12,000 scored points; most configs < 1 s/replicate, d = 2048 with N up to 2·10⁵ up to ~15 s/replicate; C3–C5 similar order | 40–110 (to be replaced by the C0 seconds-per-replicate table before C1 launches) | 160 |
| D | D1 1,000 configs, D2 3,000 + CMA-ES 1,280 evaluations × 16 replicates of NN distances (≤ 20k × N); D3 27 cells × 50 subsamples | 6–15 | 24 |
| E | betting CS on 1,001-point grid, 10,000 replicates × 2,000 groups per cell (CPU or GPU vectorised) | 1–3 | 6 |
| F | F1–F3 CPU; F4 feature extraction 3 backbones × 12 corruption settings × 20,000 patches | 4–10 | 20 |
| G | G1 frozen ≤ 5; G2 24 fine-tunes (first-run measurement replaces this) | 45–70 | 80 |
| H | H1 twins of the A1 pipeline, H3 1,000 datasets per design cell | 2–6 | 10 |

Item 0 actual: SLURM job 65255, 1 × A100, 54:52 wall clock from `sacct` (most of it the CPU sklearn reference in the kNN equality check while the GPU was held) = 0.91 GPU-h, within the cap of 1.
