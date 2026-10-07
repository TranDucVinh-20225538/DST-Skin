#!/usr/bin/env python3
"""R3 item 5b Step A (results/r3/5b/PRECOMMIT.json): pooling check on cached Track A scores, no refits.

Per FM x seed x scorer (Mahalanobis, kNN), from outputs/rigor_pack/miccai_campaign/scores/camelyon_{fm}[_s43|_s44].npz
(2-fold fits; fi = fold of each id_val patch, -1 = no fold):
  F1         = mean_g AUROC(id_val of fold 1-g scored by fit g, OOD scored by fit g)
  F2_naive   = AUROC(all id_val, each scored by the fit that did not see its slide; OOD = mean of the two fits)
  F2_matched = sum_k n_k AUROC_k / sum_k n_k, arm k = id_val of fold k with OOD, both scored by fit 1-k
  s.d. of single-fit OOD scores (fit 0, fit 1) and of the fit-averaged OOD scores
Seed 42 F1 / F2_naive are checked against results/r3/5/a_camelyon_fm.csv (1e-9).
Writes results/r3/5b/stepA.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.benchmark_metrics import calc_auroc  # noqa: E402

FMS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5")
SEEDS = (42, 43, 44)
SCORERS = ("Mahalanobis", "kNN")
SC = REPO / "outputs/rigor_pack/miccai_campaign/scores"
OUT = REPO / "results/r3/5b"


def main() -> int:
    prev = pd.read_csv(REPO / "results/r3/5/a_camelyon_fm.csv").set_index(["fm", "scorer"])
    rows = []
    for fm in FMS:
        for s in SEEDS:
            z = np.load(SC / ("camelyon_%s%s.npz" % (fm, "" if s == 42 else "_s%d" % s)))
            fi = z["fi"]
            keep = fi >= 0
            for m in SCORERS:
                id_by = {g: z["f%d_id_%s" % (g, m)] for g in (0, 1)}
                ood_by = {g: z["f%d_ood_%s" % (g, m)] for g in (0, 1)}
                F1 = float(np.mean([calc_auroc(id_by[g][fi == 1 - g], ood_by[g]) for g in (0, 1)]))
                idc = np.full(len(fi), np.nan)
                for k in (0, 1):
                    idc[fi == k] = id_by[1 - k][fi == k]
                ood_avg = (ood_by[0] + ood_by[1]) / 2
                F2n = float(calc_auroc(idc[keep], ood_avg))
                n = {k: int((fi == k).sum()) for k in (0, 1)}
                arm = {k: float(calc_auroc(id_by[1 - k][fi == k], ood_by[1 - k])) for k in (0, 1)}
                F2m = (n[0] * arm[0] + n[1] * arm[1]) / (n[0] + n[1])
                leaky = float(calc_auroc(z["std_id_%s" % m][keep], z["std_ood_%s" % m]))
                if s == 42:
                    p = prev.loc[(fm, m)]
                    assert abs(F1 - p.auroc_F1) < 1e-9 and abs(F2n - p.auroc_F2) < 1e-9, (fm, m)
                rows.append({"fm": fm, "seed": s, "scorer": m, "auroc_leaky": leaky, "F1": F1, "F2_naive": F2n,
                             "F2_matched": F2m, "F2_naive_minus_F1": F2n - F1, "F2_matched_minus_F1": F2m - F1,
                             "arm0_auroc": arm[0], "arm1_auroc": arm[1], "n_id_fold0": n[0], "n_id_fold1": n[1],
                             "sd_ood_fit0": float(np.std(ood_by[0])), "sd_ood_fit1": float(np.std(ood_by[1])),
                             "sd_ood_avg": float(np.std(ood_avg)),
                             "corr_ood_fit0_fit1": float(np.corrcoef(ood_by[0], ood_by[1])[0, 1]),
                             "sd_id_crossfit": float(np.std(idc[keep]))})
    D = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT / "stepA.csv", index=False)
    print(D.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
