"""Shared helpers for WORK ORDER T1 (v2.5): paths, hashes, seeds, cell loading, package fold maps, Track A scorers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HOME = Path.home()
REPO = HOME / "DST-Skin"
RES = REPO / "results" / "t1"
W1 = HOME / "r3work" / "item1"
INPUTS = W1 / "inputs"
CROSSFIT_DIR = HOME / "handoff" / "crossfit-ood_v3"
BUNDLE = HOME / "handoff" / "t1" / "bundle" / "t1_bundle"
MASTER_SEED = 20261010
ITEM_CODE = {"0": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9}
FMS = ["uni", "virchow2", "dinov2_vitb14", "dinov2_vitl14", "conch_v1_5"]

if str(CROSSFIT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CROSSFIT_DIR / "src"))


def precommit_hash() -> str:
    """Commit hash of the T1 precommit (first commit that added results/t1/PRECOMMIT_T1.json)."""
    out = subprocess.run(["git", "-C", str(REPO), "log", "--diff-filter=A", "--format=%H", "--",
                          "results/t1/PRECOMMIT_T1.json"], capture_output=True, text=True).stdout.split()
    return out[-1] if out else "NA"


def head_hash() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(x) -> int:
    s = x if isinstance(x, str) else json.dumps(x, sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "big")


def seed_seq(item: str, config, replicate: int = 0) -> np.random.SeedSequence:
    return np.random.SeedSequence([MASTER_SEED, ITEM_CODE[item], stable_hash(config), replicate])


def sign0(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def set_float64_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def load_cell(cell: str, keys=None) -> dict:
    z = np.load(INPUTS / f"{cell}.npz", allow_pickle=False)
    keys = z.files if keys is None else [k for k in keys if k in z.files]
    return {k: z[k] for k in keys}


def paper2fold_inputs(d: dict, seed: int):
    """Replicates crossfit_auroc's paper_2fold preprocessing (strict seen set, default of recompute_paper_ci):
    drops ID-eval samples whose group is not in train, builds the fold maps from fold_train if present,
    else with the package's own _new_fold_maps(seed). Returns (xtr, gtr, xid, gid, xood, fold_maps)."""
    from crossfit_ood import core as C
    xtr = np.asarray(d["features_train"], dtype=np.float64)
    gtr = np.asarray(d["groups_train"])
    xid = np.asarray(d["features_id_eval"], dtype=np.float64)
    gid = np.asarray(d["groups_id_eval"])
    xood = np.asarray(d["features_ood"], dtype=np.float64)
    used = np.isin(gid, gtr)
    xid, gid = xid[used], gid[used]
    if d.get("fold_train") is not None:
        fold_maps = C._fold_ids_to_maps(np.asarray(d["fold_train"]).astype(int), gtr)
    else:
        fold_maps = C._new_fold_maps("paper_2fold", gtr, gid, 2, 1, seed)
    return xtr, gtr, xid, gid, xood, fold_maps


def paper2fold_delta(scorer, xtr, gtr, xid, gid, xood, fold_maps, ytr=None, seed=0):
    """Point estimate (AUROC_seen, AUROC_unseen, Delta) through the package's own _fit_scorer."""
    from crossfit_ood import core as C
    sf = C._fit_scorer(scorer, xtr, gtr, ytr, xid, gid, xood, "paper_2fold", fold_maps, False, 0.95, seed + 2)
    a_l, a_x = sf.point(len(xid), len(xood))
    return float(a_l), float(a_x), float(a_l - a_x)
