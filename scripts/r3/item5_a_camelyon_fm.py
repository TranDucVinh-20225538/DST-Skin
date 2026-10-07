#!/usr/bin/env python3
"""R3 item 5(a), post-hoc: Camelyon17 FM leaky vs Track E F2 vs F1 per fit scorer (no refit).

Reuses the fm_gate_score.py cells (outputs/reports/rigor_pack/foundation_gate/cells/camelyon_{fm}.json) and their
per-image score caches (outputs/rigor_pack/miccai_campaign/scores/camelyon_{fm}.npz), seed 42.
- leaky = 'standard': scorer + probe fitted on all training slides, ID = id_val patches with an H11c fold.
- F1 = disjoint_2fold (fit fold f, ID = id_val of fold 1-f, fold mean)       [camp_report / camp_medbench_cpu]
- F2 = cross-fit: id_val of fold g scored by the fold-(1-g) fit, OOD averaged over the two fits
                                                                             [camp_report.cam_cache_metrics]
n_groups_fit = training slides per H11c fold (folds() seed 42). No CI: Track E code computes none.
Writes results/r3/5/a_camelyon_fm.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO))
import common as C  # noqa: E402
from leakfree_fit_scores import REPRO_FOLD_SIZES, folds  # noqa: E402
from src.utils.benchmark_metrics import calc_auroc  # noqa: E402

FMS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5")
FIT = ("Mahalanobis", "kNN", "ViM", "ReAct")
BAR = 0.02
OUT = REPO / "results/r3/5"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = C.load_camelyon_metadata(REPO)
    prev = pd.read_csv(REPO / "outputs/reports/rigor_pack/miccai_campaign/trackE_f2_medical/f2_vs_f1.csv")
    prev = prev[prev.dataset == "camelyon17"].set_index("backbone")
    rows = []
    for fm in FMS:
        j = json.loads((REPO / "outputs/reports/rigor_pack/foundation_gate/cells" / ("camelyon_%s.json" % fm)).read_text())
        z = np.load(REPO / "outputs/rigor_pack/miccai_campaign/scores" / ("camelyon_%s.npz" % fm))
        f = np.load(REPO / "outputs/rigor_pack/foundation_gate/feats" / ("camelyon_%s.npz" % fm))
        tsl, vsl = np.asarray(f["train_slide"]), np.asarray(f["id_slide"])
        tf, fi_chk = folds(tsl, meta.center.to_numpy()[np.asarray(f["train_idx"])], vsl, 42)
        assert ((tf == 0).sum(), (tf == 1).sum()) == REPRO_FOLD_SIZES
        fi = z["fi"]
        assert np.array_equal(fi, fi_chk)
        keep = fi >= 0
        n_slides = [int(len(np.unique(tsl[tf == k]))) for k in (0, 1)]
        for m in FIT:
            leaky = float(calc_auroc(z["std_id_%s" % m][keep], z["std_ood_%s" % m]))
            idc = np.full(len(fi), np.nan)
            for g in (0, 1):
                idc[fi == g] = z["f%d_id_%s" % (1 - g, m)][fi == g]
            F2 = float(calc_auroc(idc[keep], (z["f0_ood_%s" % m] + z["f1_ood_%s" % m]) / 2))
            F1 = float(np.mean([calc_auroc(z["f%d_id_%s" % (g, m)][fi == 1 - g], z["f%d_ood_%s" % (g, m)])
                                for g in (0, 1)]))
            assert abs(F1 - j["disjoint_2fold"][m]) < 1e-9, (fm, m, F1, j["disjoint_2fold"][m])
            assert abs(F2 - prev.loc[fm, "F2_%s" % m]) < 1e-9, (fm, m)
            d = abs(F2 - F1)
            rows.append({"fm": fm, "scorer": m, "auroc_leaky": leaky, "auroc_F2": F2, "auroc_F1": F1,
                         "abs_F2_minus_F1": d, "pass": bool(d <= BAR), "n_groups_fit": "%d/%d" % tuple(n_slides),
                         "K": 2, "d": int(j["feat_dim"]), "id_acc": float(j["probe"]["id_acc"]), "ci95": "n/a"})
    pd.DataFrame(rows).to_csv(OUT / "a_camelyon_fm.csv", index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
