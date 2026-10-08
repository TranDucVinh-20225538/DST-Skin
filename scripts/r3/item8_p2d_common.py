"""R3 item 8 / P2-d shared helpers (results/r3/8/p2d/PRECOMMIT.json).

Primary OOD candidate per dataset = results/r3/8/p2d/pilot_ood.json (pilot rule). Sets per (dataset, variant) come from
the Phase 0 tables (kvasir_sets.csv.gz: K_lit roles `role_lit`, K_seg roles `role_seg`; brain_sets.csv.gz: record-wise
`role`). Image key = Kvasir frame file stem / brain .mat index; staged PNG = {key}.png in data/staged/p2d_*_256.
Only the primary candidate is trained and scored (the PRECOMMIT GPU estimate covers one candidate per dataset).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P2D = REPO / "results/r3/8/p2d"
FEAT = REPO / "outputs/rigor_pack/r3_item8/p2d"
STAGE = {"kvasir": REPO / "data/staged/p2d_kvasir_256", "brain": REPO / "data/staged/p2d_brain_256"}
VARIANTS = {"kvasir": ("lit", "seg"), "brain": ("rec",)}
PILOT_KEY = {"kvasir": "kvasir_capsule", "brain": "brain"}
CNN_RUNS = [(a, s) for a in ("resnet50", "convnext_tiny") for s in (42, 43)]
FMS = ("dinov2_vitb14", "biomedclip")
RUNS = [f"{a}_s{s}" for a, s in CNN_RUNS] + list(FMS)
SCORERS = ("mahalanobis_l2", "knn_mean_cosine")


def primary(ds):
    return json.loads((P2D / "pilot_ood.json").read_text())[PILOT_KEY[ds]]["primary"]


def candidate_table(ds):
    """All rows of the primary candidate with key, grp, label and the variant role / fold columns."""
    c = primary(ds)
    if c is None:
        return None, None
    if ds == "kvasir":
        t = pd.read_csv(P2D / "kvasir_sets.csv.gz", dtype={"file": str, "video": str})
        t = t[t.ood_candidate == c].copy()
        t["key"], t["grp"] = t.file, t.video
        im = pd.read_csv(P2D / "kvasir_images.csv.gz", dtype={"file": str}).drop_duplicates("file").set_index("file")
        t["folder"] = im.folder.loc[t.file].values
    else:
        t = pd.read_csv(P2D / "brain_sets.csv.gz")
        t = t[t.ood_candidate == c].copy()
        t["key"], t["grp"], t["label"] = t.idx.astype(str), t.pid.astype(str), t.cls
        t["role_rec"], t["afit_fold_rec"] = t.role, t.afit_fold
    return t.reset_index(drop=True), c


def variant_sets(ds, variant):
    """dict role -> DataFrame (fit, ckpt, seen, unseen, ood), ID classes, candidate, fold map (fit group -> 0/1)."""
    t, c = candidate_table(ds)
    if t is None:
        return None
    role = t[f"role_{variant}"]
    S = {r: t[role == r].reset_index(drop=True) for r in ("fit", "ckpt", "seen", "unseen", "ood")}
    classes = sorted(set(S["fit"].label))
    assert set(S["ckpt"].label) <= set(classes) and set(S["seen"].label) <= set(classes), "ID class outside fit"
    fm = S["fit"].groupby("grp")[f"afit_fold_{variant}"].agg(["min", "max"])
    assert (fm["min"] == fm["max"]).all() and fm["min"].notna().all(), "afit fold not constant per fit group"
    return {"sets": S, "classes": classes, "candidate": c, "fold": fm["min"].astype(int).to_dict()}


def macro_ovr_auroc(y, P, classes):
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, c in enumerate(classes):
        yy = (np.asarray(y) == c).astype(int)
        if 0 < yy.sum() < len(yy):
            per[c] = float(roc_auc_score(yy, P[:, j]))
    return float(np.mean(list(per.values()))), per
