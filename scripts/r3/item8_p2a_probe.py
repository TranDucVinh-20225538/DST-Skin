#!/usr/bin/env python3
"""R3 item 8 / P2-a FM competence gate (results/r3/8/PRECOMMIT.json, G_competence): linear probe on frozen features,
fm_gate_score.probe recipe (fm_ood_pilot.fit_linear_head: LogisticRegression(C=1.0, lbfgs, max_iter=2000) on the raw
features), one-vs-rest per finding (14), fitted on the fit images; macro AUROC on seen (gate: >= 0.70) and unseen.
Writes results/r3/8/p2a/train/{fm}_probe.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def main() -> int:
    fm = sys.argv[1]
    t0 = time.time()
    z = np.load(REPO / "outputs/rigor_pack/r3_item8/p2a" / f"{fm}.npz", allow_pickle=True)
    Xf, Yf = np.asarray(z["fit_feats"], dtype=np.float64), np.asarray(z["fit_labels"])
    per = {"seen": {}, "unseen": {}}
    for j, f in enumerate(FINDINGS):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(Xf, Yf[:, j])
        for k in per:
            y = np.asarray(z[f"{k}_labels"])[:, j]
            if 0 < y.sum() < len(y):
                per[k][f] = float(roc_auc_score(y, clf.decision_function(np.asarray(z[f"{k}_feats"], dtype=np.float64))))
        print(f, {k: round(v.get(f, float("nan")), 4) for k, v in per.items()}, round(time.time() - t0), flush=True)
    macro = {k: float(np.mean(list(v.values()))) for k, v in per.items()}
    res = {"fm": fm, "probe": "LogisticRegression(C=1.0, lbfgs, max_iter=2000) per finding, raw features",
           "macro_auroc": macro, "per_finding_auroc": per, "G_competence": {"probe_seen_macro_auroc_ge_0.70": macro["seen"] >= 0.70,
                                                                           "pass": macro["seen"] >= 0.70},
           "seconds": round(time.time() - t0, 1)}
    (REPO / "results/r3/8/p2a/train" / f"{fm}_probe.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "per_finding_auroc"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
