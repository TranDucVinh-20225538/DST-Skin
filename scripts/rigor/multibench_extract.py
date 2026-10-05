#!/usr/bin/env python3
"""Indexed features of a standard (full-train, seed 42) model for multibench (A)/(C).

Same extractor as the pilots (avgpool hook, eval transform), but loaders keep the WILDS split order
so every row carries its dataset index, group and fold. Checkpoint reused unchanged.
Output: outputs/rigor_pack/multibench/{ds}/{arch}_s42_full.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multibench_common as M  # noqa: E402


def extract(model, dset, ds, arch, bs, device, out: Path, splits=("train", "id", "ood")):
    from src.models.cnn_family import fc_params
    from src.utils.feature_extractor import extract_features_and_logits
    sp = M.SPECS[ds]
    fold, g = M.folds(dset, ds)
    y = np.asarray(dset.y_array).astype(np.int64)
    tf = M.transform(ds, arch, train=False)
    arr = {}
    for key in splits:
        idx = M.split_idx(dset, {"train": "train", "id": sp["id"], "ood": sp["ood"]}[key])
        lo, fe = extract_features_and_logits(model, M.loader(dset, idx, tf, bs, False), device)
        arr.update({key + "_logits": lo, key + "_feats": fe, key + "_labels": y[idx], key + "_idx": idx,
                    key + "_group": g[idx], key + "_fold": fold[idx]})
    w, b = fc_params(model)
    np.savez(out, fc_weight=np.asarray(w), fc_bias=np.asarray(b), **arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(M.SPECS))
    ap.add_argument("--arch", required=True)
    args = ap.parse_args()
    import torch
    from src.models.cnn_family import get_cnn_backbone, recipe

    out = M.feat_dir(args.ds) / ("%s_s42_full.npz" % args.arch)
    if out.exists():
        print("exists", out)
        return 0
    t0 = time.time()
    dset = M.get_dataset(args.ds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_cnn_backbone(args.arch, num_classes=M.SPECS[args.ds]["n_classes"], pretrained=False)
    model.load_state_dict(torch.load(M.published_ckpt(args.ds, args.arch), map_location=device))
    model.to(device)
    extract(model, dset, args.ds, args.arch, int(recipe(args.arch)["batch_size"]), device, out)
    print("saved %s in %.0f s" % (out.name, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
