#!/usr/bin/env python3
"""R3 item 1 stop-check: ID-eval ("seen") samples whose group has a Track A fold (medbench fold_map, Phase-0 split
fold column) but no training image. Track A counts them as same-fold; the package paper_2fold drops them
(seen = group present in the fit fold). Writes results/r3/1/orphan_seen.csv (one row per medical cell)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/rigor")]
import medbench_scores as M  # noqa: E402

W = Path.home() / "r3work/item1"
rows, cache = [], {}
for p in ("med", "medfm"):
    for c in pd.read_csv(W / f"cells_{p}.csv").itertuples():
        z = np.load(W / "inputs" / f"{c.cell}.npz", allow_pickle=True)
        gtr, gid = z["groups_train"].astype(str), z["groups_id_eval"].astype(str)
        key = (c.dataset, len(gtr), len(gid), hash(gtr.tobytes()), hash(gid.tobytes()))
        if key not in cache:
            src = "foundation_gate" if c.kind == "fm" else "medbench"
            tag = c.cell[len(c.dataset) + 1:]
            labels = np.load(REPO / f"outputs/rigor_pack/{src}/{c.dataset}/{tag}.npz")["train_labels"]
            fm = M.fold_map(c.dataset, gtr, labels)
            fs = np.array([fm.get(g, -1) for g in gid])
            orphan = (fs >= 0) & ~np.isin(gid, gtr)
            cache[key] = dict(n_id_eval=len(gid), n_trackA_counted=int((fs >= 0).sum()),
                              n_orphan=int(orphan.sum()), orphan_groups=int(len(np.unique(gid[orphan]))),
                              frac_orphan=float(orphan.mean()))
        rows.append(dict(cell=c.cell, dataset=c.dataset, **cache[key]))
d = pd.DataFrame(rows)
(REPO / "results/r3/1").mkdir(parents=True, exist_ok=True)
d.to_csv(REPO / "results/r3/1/orphan_seen.csv", index=False)
print(d.groupby("dataset")[["n_id_eval", "n_orphan", "orphan_groups", "frac_orphan"]].agg(["min", "max"]).to_string())
