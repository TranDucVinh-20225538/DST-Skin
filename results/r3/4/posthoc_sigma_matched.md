# R3 item 4: post-hoc check of the sigma_b explanation (exact / LW Mahalanobis)

commit: 37c3ea0

Post-hoc check, not in PRECOMMIT.json; the precommitted verdict in REPORT.md is unchanged. Both placements use the same sigma_b (the calibrated high- or low-placement value for that d); everything else as in item 4 (paper_2fold, K = 2, n_groups_fit = 15 per fold, 50 paired replicates). ID accuracy: not applicable (synthetic); probe columns = slide-ID probe balanced accuracy at the matched sigma_b.

| scorer | sigma_b from | sigma_b | d | n/d | Delta_high | Delta_low | D = high - low | 95% CI | probe high / low |
|---|---|---|---|---|---|---|---|---|---|
| exact_maha | high | 1.0938 | 768 | 1 | +0.0284 | +0.0196 | +0.0088 | [+0.0046, +0.0129] | 0.896 / 0.720 |
| exact_maha | high | 1.0938 | 768 | 10 | +0.0430 | +0.0434 | -0.0003 | [-0.0007, +0.0000] | 0.896 / 0.720 |
| exact_maha | high | 1.0625 | 2560 | 1 | +0.0164 | +0.0130 | +0.0034 | [+0.0015, +0.0053] | 0.904 / 0.343 |
| exact_maha | high | 1.0625 | 2560 | 10 | +0.0213 | +0.0214 | -0.0000 | [-0.0001, +0.0001] | 0.904 / 0.343 |
| exact_maha | low | 1.4375 | 768 | 1 | +0.0488 | +0.0354 | +0.0134 | [+0.0078, +0.0190] | 0.979 / 0.897 |
| exact_maha | low | 1.4375 | 768 | 10 | +0.0766 | +0.0776 | -0.0009 | [-0.0015, -0.0004] | 0.979 / 0.897 |
| exact_maha | low | 2.2500 | 2560 | 1 | +0.0615 | +0.0462 | +0.0153 | [+0.0110, +0.0196] | 0.999 / 0.896 |
| exact_maha | low | 2.2500 | 2560 | 10 | +0.0960 | +0.0964 | -0.0004 | [-0.0007, -0.0001] | 0.999 / 0.896 |
| lw_maha | high | 1.0938 | 768 | 1 | +0.0526 | +0.0246 | +0.0279 | [+0.0246, +0.0313] | 0.896 / 0.720 |
| lw_maha | high | 1.0938 | 768 | 10 | +0.0438 | +0.0428 | +0.0010 | [+0.0006, +0.0014] | 0.896 / 0.720 |
| lw_maha | high | 1.0625 | 2560 | 1 | +0.0260 | +0.0170 | +0.0090 | [+0.0079, +0.0101] | 0.904 / 0.343 |
| lw_maha | high | 1.0625 | 2560 | 10 | +0.0215 | +0.0212 | +0.0003 | [+0.0001, +0.0004] | 0.904 / 0.343 |
| lw_maha | low | 1.4375 | 768 | 1 | +0.0894 | +0.0464 | +0.0430 | [+0.0385, +0.0476] | 0.979 / 0.897 |
| lw_maha | low | 1.4375 | 768 | 10 | +0.0779 | +0.0767 | +0.0012 | [+0.0007, +0.0018] | 0.979 / 0.897 |
| lw_maha | low | 2.2500 | 2560 | 1 | +0.0909 | +0.0645 | +0.0264 | [+0.0227, +0.0301] | 0.999 / 0.896 |
| lw_maha | low | 2.2500 | 2560 | 10 | +0.0966 | +0.0961 | +0.0005 | [+0.0002, +0.0009] | 0.999 / 0.896 |
