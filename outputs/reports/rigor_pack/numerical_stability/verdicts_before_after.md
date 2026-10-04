| H | depends on anchor Maha/ViM | as run (precommit pipeline) | old code, 10 repeats | stable (Maha LW64 + ViM 90%) | final (addendum rule) | extra: stable Maha + ViM 8 threads | extra: stable Maha, no ViM |
|---|---|---|---|---|---|---|---|
| H1 | yes | mixed | arch separable (7/10); mixed (3/10) | mixed | INCONCLUSIVE (numerical instability) | mixed | mixed |
| H2 | yes | all above chance | all above chance (10/10) | all above chance | all above chance | all above chance | all above chance |
| H3 | yes | D>0 (cluster CI excludes 0); sampling not what moves W | D>0 (cluster CI excludes 0); sampling not what moves W (10/10) | D>0 (cluster CI excludes 0); sampling not what moves W | D>0 (cluster CI excludes 0); sampling not what moves W | D>0 (cluster CI excludes 0); sampling not what moves W | D>0 (cluster CI excludes 0); sampling not what moves W |
| H4 | yes | arch effect detected | arch effect detected (10/10) | arch effect detected | arch effect detected | arch effect detected | arch effect detected |
| H5 | yes | (a) every backbone and seed; (b) pass | (a) every backbone and seed; (b) pass (10/10) | (a) downgraded to 'usually'; (b) fail | INCONCLUSIVE (numerical instability) | (a) every backbone and seed; (b) pass | (a) every backbone and seed; (b) pass |
| H6 | no | supported | - | - | supported | - | - |
| H7 | yes | not AUROC-specific | not AUROC-specific (10/10) | changes under some metric | INCONCLUSIVE (numerical instability) | not AUROC-specific | not AUROC-specific |
| H8 | no | T-invariance check FAIL (max |dAUROC| 0.17); (b) jump not explained by accuracy | - | - | T-invariance check FAIL (max |dAUROC| 0.17); (b) jump not explained by accuracy | - | - |
| H9 | yes | risk-level dependent; triage-score robust | risk-level dependent; triage-score robust (10/10) | risk-level dependent; triage-score robust | risk-level dependent; triage-score robust | risk-level dependent; triage-score robust | risk-level dependent; triage-score robust |
| H10 | no | gap kept (cluster CI excludes 0); per-seed MSP-jump cluster CI not produced by persample.py | - | - | gap kept (cluster CI excludes 0); per-seed MSP-jump cluster CI not produced by persample.py | - | - |
| H11 | no | (a,b) pass; (c) pending (leak job) | - | - | (a,b) pass; (c) pending (leak job) | - | - |
| H12 | no | skin_isic_pad: H1 mixed, H4 not detected; midog: H1 mixed, H4 arch effect detected | - | - | skin_isic_pad: H1 mixed, H4 not detected; midog: H1 mixed, H4 arch effect detected | - | - |
| H13 | no | seed-robust locked prediction | - | - | seed-robust locked prediction | - | - |
| H14 | yes | descriptive (Holm within family) | descriptive (Holm within family) (10/10) | descriptive (Holm within family) | descriptive; see families | descriptive (Holm within family) | descriptive (Holm within family) |
| H15 | no | pending (recipe seeds incomplete) | - | - | pending (recipe seeds incomplete) | - | - |
| H16 | yes | 0.185 kept with percentile; feature regret >0.05 (fraction reported) | 0.185 kept with percentile; feature regret >0.05 (fraction reported) (10/10) | 0.185 = worst case wording; feature regret >0.05 (fraction reported) | INCONCLUSIVE (numerical instability) | 0.185 kept with percentile; feature regret >0.05 (fraction reported) | 0.185 kept with percentile; feature regret >0.05 (fraction reported) |
| H17 | yes | seeds agree more (D>0.10) | seeds agree more (D>0.10) (10/10) | tie-robust (|D|<=0.10) | INCONCLUSIVE (numerical instability) | seeds agree more (D>0.10) | tie-robust (|D|<=0.10) |
