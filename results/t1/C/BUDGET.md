# Item C — budget from the C0 throughput table (replaces the planning estimate, rule 16)

Measured on NVIDIA A100-SXM4-80GB (C0, `c0_seconds_per_rep.csv`, 20 (d, G, n) points, d 16–2048, N 1,200–200,000):
seconds per replicate 0.35–5.4 s. Fit used for the projection (crude; max relative error 80 %):
s/rep ≈ 0.63 − 1.32·N d²/10¹² + 14.3·N d/10⁹.

Projection over the committed C work list (`PRECOMMIT_T1_addendum_C.json`), at the two ends of the adaptive replicate
rule (8 and 64 replicates per config):

| block | units | GPU-h at R = 8 | GPU-h at R = 64 |
|---|---|---|---|
| C1 | 6,000 | 9.5 | 76.1 |
| C3 | 378 | 0.5 | 4.2 |
| C4 (scorer cap, c4sc) | 300 | 0.4 | 3.3 |
| C4 (TV oracle, c4tv) | 75 | grid computation, not covered by the fit (< 0.5 expected) | |
| C5 (stage 3) | 7,500 | 11.7 | 93.3 |

C-core (C1, C3, C4) projected 10–84 GPU-h; with C5 22–177 GPU-h. Hard cap 160 GPU-h (rule 16), counted from the
sum of the per-unit seconds ledgers (`raw/*.secs`, C0 included); when it is reached the run stops and the fraction
completed is reported. C runs with at most 2 GPUs at a time (campaign limit of 4 GPUs, DEVIATIONS D0-5).
