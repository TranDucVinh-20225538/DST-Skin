#!/usr/bin/env python3
"""R3 item 8 (post hoc sensitivity, results/r3/8/ood_overlap/audit.json): Track A Kermany (arm std) R3 item-1 inputs
with the OOD set restricted to images whose patient is absent from every ID set of the arm (train, val, test_seen,
test_unseen, unseen_extra). Everything else is copied unchanged from ~/r3work/item1/inputs/<cell>.npz.
Writes ~/r3work/item8_oodsens/inputs/<cell>.npz and list_kermany_std.txt (inputs are not committed).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
from medbench_scores import groups_of  # noqa: E402

SRC, DST = Path.home() / "r3work/item1", Path.home() / "r3work/item8_oodsens"


def main():
    d = pd.read_csv(REPO / "outputs/reports/rigor_pack/medbench/splits/kermany.csv.gz", low_memory=False)
    v2 = d[d.version == "v2"]
    id_g = set(v2.group[v2.role_std.isin(["train", "val", "test_seen", "test_unseen"])].astype(str)) | \
        set(d.group[d.role_std == "unseen_extra"].astype(str))
    n_clean = json.loads((REPO / "results/r3/8/ood_overlap/audit.json").read_text())["kermany_std"]["clean_ood_images"]
    cells = [r["cell"] for r in csv.DictReader(open(SRC / "cells_med.csv")) if r["dataset"] == "kermany"]
    (DST / "inputs").mkdir(parents=True, exist_ok=True)
    for c in cells:
        z = dict(np.load(SRC / "inputs" / f"{c}.npz"))
        f = np.load(REPO / "outputs/rigor_pack/medbench/kermany" / f"{c[len('kermany_'):]}.npz")
        assert np.array_equal(z["features_ood"], f["ood_feats"].astype(np.float32)), c
        keep = ~np.isin(groups_of("kermany", f["ood_keys"]), list(id_g))
        assert int(keep.sum()) == n_clean, (c, int(keep.sum()), n_clean)
        z["features_ood"] = z["features_ood"][keep]
        np.savez(DST / "inputs" / f"{c}.npz", **z)
        print(c, len(keep), int(keep.sum()), flush=True)
    (DST / "list_kermany_std.txt").write_text("\n".join(cells) + "\n")


if __name__ == "__main__":
    main()
