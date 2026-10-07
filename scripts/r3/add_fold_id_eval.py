#!/usr/bin/env python3
"""Add fold_id_eval (Track A fold per ID-eval sample: medbench fold_map(...).get(group, -1)) to every medical
R3 item-1 input .npz, for recompute_paper_ci.py --seen tracka. Camelyon inputs need none (every ID-eval slide is
a training slide). Checks that fold_id_eval agrees with fold_train on train groups."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/rigor")]
import medbench_scores as M  # noqa: E402

W = Path.home() / "r3work/item1"
for p in ("med", "medfm"):
    for c in pd.read_csv(W / f"cells_{p}.csv").itertuples():
        f = W / "inputs" / f"{c.cell}.npz"
        z = dict(np.load(f, allow_pickle=True))
        labels = np.load(REPO / f"outputs/rigor_pack/medbench/{c.dataset}/{c.cell[len(c.dataset) + 1:]}.npz")["train_labels"]
        gtr, gid = z["groups_train"].astype(str), z["groups_id_eval"].astype(str)
        fm = M.fold_map(c.dataset, gtr, labels)
        fe = np.array([fm.get(g, -1) for g in gid], dtype=np.int8)
        tmap = dict(zip(gtr, z["fold_train"]))
        known = np.array([tmap.get(g, -1) for g in gid])
        assert np.all((known < 0) | (known == fe)), c.cell
        z["fold_id_eval"] = fe
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, **z)
        tmp.replace(f)
        print(c.cell, "orphan", int(((known < 0) & (fe >= 0)).sum()), "dropped", int((fe < 0).sum()), flush=True)
