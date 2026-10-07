#!/usr/bin/env python3
"""R3 item 1: add the Track A fold assignment (and missing heads) to every item-1 input.

  medbench (CNN and FM probes): fold per train sample = medbench_scores.fold_map(ds, groups, train_labels)
  Camelyon: H11c folds = leakfree_fit_scores.folds(train_slide, hospital, id_slide, seed) with the cell seed
            (fm_gate_score.py uses --seed as the fold seed); FM heads = fm_gate_score.probe on all training
            embeddings (binary [-W, W], [-b, b]).
Rewrites <work>/inputs/<cell>.npz with key fold_train, writes <cell>_head.npz for Camelyon FMs and
<work>/vim_dim.csv (cell, vim_dim = D - C, the Track A fit_vim default).

    python add_tracka_folds.py --part med|camelyon [--cells c1 c2 ...]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
W = Path.home() / "r3work/item1"
FMS = ("uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5")


def rewrite(cell, extra):
    p = W / "inputs" / (cell + ".npz")
    z = dict(np.load(p, allow_pickle=True))
    z.update(extra)
    tmp = p.with_suffix(".tmp.npz")
    np.savez(tmp, **z)
    tmp.replace(p)
    return z


def med(cell):
    from medbench_scores import fold_map
    ds, tag = cell.split("_", 1)
    z = np.load(REPO / f"outputs/rigor_pack/medbench/{ds}/{tag}.npz")
    zi = np.load(W / "inputs" / (cell + ".npz"), allow_pickle=True)
    gt = zi["groups_train"].astype(str)
    fm = fold_map(ds, gt, z["train_labels"])
    ft = np.array([fm[g] for g in gt], dtype=int)
    rewrite(cell, {"fold_train": ft})
    return int(zi["features_train"].shape[1]) - int(z["fc_weight"].shape[0])


def camelyon(cell):
    import common as C
    from leakfree_fit_scores import folds
    m = re.match(r"camelyon_(.+)_s(\d+)$", cell)
    model, seed = m.group(1), int(m.group(2))
    meta = C.load_camelyon_metadata(REPO)
    if model in FMS:
        z = np.load(REPO / "outputs/rigor_pack/foundation_gate/feats" / ("camelyon_%s.npz" % model))
        g = lambda k: np.asarray(z[k])  # noqa: E731
        vk = "id_slide"
    else:
        import torch
        d = torch.load(C.feature_path(REPO, "camelyon17", model, seed, indexed=True), map_location="cpu",
                       weights_only=False)
        g = lambda k: np.asarray(d[k].numpy() if hasattr(d[k], "numpy") else d[k])  # noqa: E731
        vk = "val_slide"
    tf, _ = folds(g("train_slide"), meta.center.to_numpy()[g("train_idx")], g(vk), seed)
    zi = np.load(W / "inputs" / (cell + ".npz"), allow_pickle=True)
    if not np.array_equal(zi["groups_train"].astype(str), g("train_slide").astype(str)):
        raise SystemExit("STOP: train order differs for %s" % cell)
    rewrite(cell, {"fold_train": tf.astype(int)})
    if model in FMS:
        from fm_gate_score import probe
        Wh, bh, _, _ = probe(np.asarray(g("train_feats"), dtype=np.float64), np.asarray(g("train_labels")))
        np.savez(W / "inputs" / (cell + "_head.npz"), weight=Wh, bias=bh)
        n_cls = Wh.shape[0]
    else:
        n_cls = int(g("fc_weight").shape[0])
    return int(zi["features_train"].shape[1]) - n_cls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("med", "camelyon"), required=True)
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    lists = {"med": ("list_med.txt", "list_medfm.txt"), "camelyon": ("list_camelyon.txt",)}[a.part]
    cells = a.cells or [c for f in lists for c in (W / f).read_text().split()]
    out = W / ("vim_dim_%s.csv" % a.part)
    done = {r["cell"]: r["vim_dim"] for r in csv.DictReader(open(out))} if out.exists() and not a.cells else {}
    for c in cells:
        if c in done:
            continue
        done[c] = (med if a.part == "med" else camelyon)(c)
        print(c, done[c], flush=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "vim_dim"])
            w.writerows(done.items())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
