#!/usr/bin/env python3
"""R3 item 1: one recompute_paper_ci.py input (.npz) per Track A feature set.

Track A A-fit sets: scorer fit on training groups, ID-eval = held-out images of training groups
(medbench: test_seen / seen; Camelyon: id_val of the 30 training slides), OOD = primary OOD set
(medbench `ood`; Camelyon hospital 2). OOD images are the resampling units (no groups_ood), as in
the preregistered bootstrap. Head (weight, bias) written next to the input when the model has one
(CNN fc or the medbench FM probe); Camelyon FMs have no saved head -> residual-only ViM.

Writes <out>/inputs/<cell>.npz, <out>/inputs/<cell>_head.npz and <out>/cells.csv
(cell, dataset, model, seed, kind, d, n_train, n_id, n_ood, n_groups_train, id_acc, head, trackA_delta_*).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO))

MED = ("dermamnist", "isic2019", "kermany", "breakhis")
FMS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5")
SCORER_KEY = {"mahalanobis": "Mahalanobis", "knn": "kNN", "vim": "ViM"}


def med_cells(fm=False):
    rp = REPO / "outputs/reports/rigor_pack" / ("foundation_gate" if fm else "medbench")
    for ds in MED:
        for p in sorted((rp / ds).glob("scores_fm_*.json" if fm else "scores_*.json")):
            r = json.loads(p.read_text())
            if "afit" not in r:
                continue
            tag = p.stem[len("scores_"):]
            yield ds, tag, r


def write_med(ds, tag, r, out):
    from medbench_scores import groups_of
    z = np.load(REPO / f"outputs/rigor_pack/medbench/{ds}/{tag}.npz")
    sets = {k[:-6] for k in z.files if k.endswith("_feats")}
    seen = ["test_seen"] if "test_seen" in sets else ["seen"]
    cat = lambda f: np.concatenate([z[f"{n}_{f}"] for n in seen])  # noqa: E731
    xid, kid, yid, lid = cat("feats"), cat("keys"), cat("labels"), cat("logits")
    gtr, gid = groups_of(ds, z["train_keys"]), groups_of(ds, kid)
    cell = f"{ds}_{tag}"
    np.savez(out / f"{cell}.npz", features_train=z["train_feats"].astype(np.float32), groups_train=gtr,
             features_id_eval=xid.astype(np.float32), groups_id_eval=gid,
             features_ood=z["ood_feats"].astype(np.float32))
    np.savez(out / f"{cell}_head.npz", weight=z["fc_weight"].astype(np.float64), bias=z["fc_bias"].astype(np.float64))
    m = re.match(r"(.+)_s(\d+)_(.+)$", tag)
    model, seed = m.group(1), int(m.group(2))
    return dict(cell=cell, dataset=ds, model=model, seed=seed, kind="fm" if model.startswith("fm_") else "cnn",
                d=int(z["train_feats"].shape[1]), n_train=int(len(gtr)), n_id=int(len(gid)),
                n_ood=int(len(z["ood_feats"])), n_groups_train=int(len(np.unique(gtr))),
                id_acc=float((lid.argmax(1) == yid).mean()), head="fc" if not model.startswith("fm_") else "probe",
                **{f"trackA_delta_{s}": r["afit"]["delta_fit"][k] for s, k in SCORER_KEY.items()})


def camelyon_cells():
    import torch
    import common as C
    cells = REPO / "outputs/reports/rigor_pack/foundation_gate/cells"
    for seed in (42, 43, 44):
        for p in sorted((REPO / f"outputs/features/camelyon17/frac1/seed{seed}/rigor_indexed").glob("*_features.pt")):
            arch = p.name[:-len("_features.pt")].replace("_224", "")
            yield "cnn", arch, seed, lambda p=p: torch.load(p, map_location="cpu", weights_only=False), cells
    for fm in FMS:
        for seed in (42, 43, 44):
            yield "fm", fm, seed, None, cells


def write_camelyon(kind, model, seed, loader, cells, out, lw64):
    cell = f"camelyon_{model}_s{seed}"
    trA = {}
    if kind == "fm":
        z = np.load(REPO / f"outputs/rigor_pack/foundation_gate/feats/camelyon_{model}.npz")
        g = lambda k: np.asarray(z[k])  # noqa: E731
        js = cells / ("camelyon_%s%s.json" % (model, "" if seed == 42 else "_s%d" % seed))
    else:
        d = loader()
        g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
        js = cells / ("camelyon_%s%s.json" % (model, "" if seed == 42 else "_s%d" % seed))
    if js.exists():
        r = json.loads(js.read_text())
        trA = {f"trackA_delta_{s}": r["delta"][k] for s, k in SCORER_KEY.items()}
        id_acc = r["probe"]["id_acc"]
    elif kind == "cnn" and model in lw64:
        trA = {"trackA_delta_mahalanobis": -lw64[model]["delta_maha_2fold"],
               "trackA_delta_knn": -lw64[model]["delta_knn_2fold"], "trackA_delta_vim": float("nan")}
        id_acc = None
    else:
        id_acc = None
    tk = "train_feats"
    ik, ok = ("val_feats", "ood_feats") if kind == "cnn" else ("id_feats", "ood_feats")
    sk = ("train_slide", "val_slide") if kind == "cnn" else ("train_slide", "id_slide")
    xtr, gtr, xid, gid = g(tk), g(sk[0]).astype(str), g(ik), g(sk[1]).astype(str)
    np.savez(out / f"{cell}.npz", features_train=xtr.astype(np.float32), groups_train=gtr,
             features_id_eval=xid.astype(np.float32), groups_id_eval=gid, features_ood=g(ok).astype(np.float32))
    head = "none (residual-only)"
    if kind == "cnn":
        np.savez(out / f"{cell}_head.npz", weight=g("fc_weight").astype(np.float64), bias=g("fc_bias").astype(np.float64))
        head = "fc"
        if id_acc is None:
            id_acc = float((g("val_logits").argmax(1) == g("val_labels")).mean())
    return dict(cell=cell, dataset="camelyon", model=model, seed=seed, kind=kind, d=int(xtr.shape[1]),
                n_train=int(len(gtr)), n_id=int(len(gid)), n_ood=int(len(g(ok))),
                n_groups_train=int(len(np.unique(gtr))), id_acc=id_acc, head=head, **trA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=["med", "medfm", "camelyon"], required=True)
    a = ap.parse_args()
    out = Path(a.out) / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.part in ("med", "medfm"):
        for ds, tag, r in med_cells(fm=a.part == "medfm"):
            rows.append(write_med(ds, tag, r, out))
            print(rows[-1]["cell"], flush=True)
    else:
        lw = REPO / "outputs/reports/rigor_pack/leakage_lw64/leakfree_knn.csv"
        lw64 = {r["arch"]: {k: float(v) for k, v in r.items() if k.startswith("delta")} for r in csv.DictReader(open(lw))}
        for kind, model, seed, loader, cells in camelyon_cells():
            rows.append(write_camelyon(kind, model, seed, loader, cells, out, lw64))
            print(rows[-1]["cell"], flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(Path(a.out) / f"cells_{a.part}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
