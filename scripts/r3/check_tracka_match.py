#!/usr/bin/env python3
"""R3 item 1 stop-check: does the package point estimate reproduce Track A Delta_fit (tolerance 0.001)?

Same call as recompute_paper_ci.py (crossfit_auroc, protocol paper_2fold) with the Track A folds (fold_train) and
two scorer sets:
  tracka : mahalanobis_l2, knn_mean_cosine (k = 50), vim = ViMScorer(head, dim = D - C)   (Track A definitions)
  package: mahalanobis, knn (k = 50), vim = ViMScorer(head, default dim)                  (as run before)
  tracka_seen: tracka scorers + fold_ids_eval (Track A seen set: ID-eval groups with a Phase-0 fold but no
             training image count as seen); medical cells only
Writes / merges <work>/match/<cell>.json. Cell = line SLURM_ARRAY_TASK_ID + 1 of $LIST (or --cells).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from crossfit_ood import ViMScorer, crossfit_auroc, get_scorer

W = Path.home() / "r3work/item1"


def vim_dims():
    out = {}
    for p in W.glob("vim_dim_*.csv"):
        out.update({r["cell"]: int(r["vim_dim"]) for r in csv.DictReader(open(p))})
    return out


def run(cell, sets):
    z = np.load(W / "inputs" / (cell + ".npz"), allow_pickle=True)
    hp = W / "inputs" / (cell + "_head.npz")
    h = np.load(hp) if hp.exists() else None
    vd = vim_dims()[cell]
    res = {"cell": cell, "vim_dim": vd, "head": h is not None}
    for name in sets:
        if name in ("tracka", "tracka_seen"):
            sc = {"mahalanobis": get_scorer("mahalanobis_l2"), "knn": get_scorer("knn_mean_cosine", k=50),
                  "vim": ViMScorer(weight=h["weight"], bias=h["bias"], dim=vd) if h is not None else ViMScorer(dim=vd)}
        else:
            sc = {"mahalanobis": get_scorer("mahalanobis"), "knn": get_scorer("knn", k=50),
                  "vim": ViMScorer(weight=h["weight"], bias=h["bias"]) if h is not None else ViMScorer()}
        fe = None
        if name == "tracka_seen":
            if "fold_id_eval" not in z.files:
                continue
            fe = z["fold_id_eval"].astype(int)
        t = time.time()
        rep = crossfit_auroc(z["features_train"], z["groups_train"], z["features_id_eval"], z["groups_id_eval"],
                             z["features_ood"], scorers=sc, protocol="paper_2fold", fold_ids=z["fold_train"].astype(int),
                             fold_ids_eval=fe, uncertainty="bootstrap", n_bootstrap=0, random_state=0)
        res[name] = {s: {"seen": r.auroc_leaky, "unseen": r.auroc_crossfit, "delta": r.delta}
                     for s, r in rep.results.items()}
        res[name + "_seconds"] = round(time.time() - t, 1)
        print(cell, name, {s: round(v["delta"], 5) for s, v in res[name].items()}, flush=True)
    (W / "match").mkdir(exist_ok=True)
    out = W / "match" / (cell + ".json")
    if out.exists():
        res = {**json.loads(out.read_text()), **res}
    out.write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*")
    ap.add_argument("--sets", nargs="+", default=["tracka", "package"])
    a = ap.parse_args()
    cells = a.cells or [Path(os.environ["LIST"]).read_text().split()[int(os.environ["SLURM_ARRAY_TASK_ID"])]]
    for c in cells:
        run(c, a.sets)


if __name__ == "__main__":
    main()
