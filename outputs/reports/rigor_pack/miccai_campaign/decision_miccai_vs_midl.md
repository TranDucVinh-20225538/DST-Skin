# Decision: MICCAI vs MIDL (L7)

**Venue: MICCAI**

Rule (precommit L7, locked before results): Δ_fit (feature) >= 0.05 on >= 2 medical datasets AND ranking flip on >= 1 medical → MICCAI; else MIDL.

| dataset | median feature Δ_fit (all backbones) | CNN median | FM median | n backbones (FM) | >= 0.05 | ranking rows changed / n | median τ-b | flip |
|---|---|---|---|---|---|---|---|---|
| camelyon17 | +0.091 | +0.108 | +0.067 | 13 (5) | True | 5 / 13 | +0.714 | False |
| breakhis | +0.151 | +0.155 | +0.078 | 13 (5) | True | 39 / 65 | +0.524 | True |
| isic2019 | +0.026 | +0.034 | +0.020 | 13 (5) | False | 12 / 31 | +0.810 | False |
| dermamnist | +0.017 | +0.021 | +0.010 | 12 (4) | False | 1 / 30 | +0.857 | False |
| kermany | +0.030 | +0.030 | NA | 8 (0) | False | 8 / 16 | +0.533 | True |

- Datasets counted toward Δ >= 0.05: camelyon17, breakhis (2)
- Camelyon 3-seed median feature Δ_fit (anchors + FMs, seeds 42-44, n = 21): +0.070 → unstable = False
- Datasets with a ranking flip: breakhis, kermany
- Secondary: ReAct median |Δ_fit| over Camelyon FM cells = +0.000
