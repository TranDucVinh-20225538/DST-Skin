"""Medbench arms (decisions/precommit_group_leakage_medbench_2026-10-06.md, L1-L3) built from the
Phase-0 split CSVs. An arm = train / val sets + named evaluation sets (ID and OOD), with staged keys
(see medbench_stage.py) and integer labels (-1 for OOD)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SPL = REPO / "outputs/reports/rigor_pack/medbench/splits"
RAW = REPO / "data/raw/medbench"

CLASSES = {
    "dermamnist": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
    "isic2019": ["MEL", "NV", "BCC", "AK", "BKL", "SCC"],
    "kermany": ["CNV", "DME", "NORMAL"],
    "breakhis": ["A", "F", "TA", "DC", "LC", "MC"],
    "brain_cheng": ["meningioma", "glioma", "pituitary"],
}
EPOCHS = {"dermamnist": 43, "isic2019": 16, "kermany": 10, "breakhis": 50, "brain_cheng": 50}
CORE = ["resnet18", "resnet50", "densenet121", "convnext_tiny"]
ZOO = ["mobilenet_v3_large", "regnet_y_3_2gf", "effb3", "efficientnet_v2_s"]
INPUT = {"effb3": 300}  # all others 224 (EfficientNetV2-S @224 per L2)


def input_size(arch: str) -> int:
    return INPUT.get(arch, 224)


def stage_size(arch: str) -> int:
    return 343 if input_size(arch) == 300 else 256


def member(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:20] + ".jpg"


def _sel(keys, labels):
    return {"keys": [str(k) for k in keys], "labels": np.asarray(labels, dtype=np.int64)}


def _lab(ds, s):
    m = {c: i for i, c in enumerate(CLASSES[ds])}
    return [m.get(x, -1) for x in s]


def _npz_n(path, split):
    with np.load(path) as z:
        return z[f"{split}_labels"].ravel().astype(int)


def arm(ds: str, name: str) -> dict:
    sets = {}
    if ds == "dermamnist":
        d = pd.read_csv(SPL / "dermamnist.csv.gz")
        y = _lab(ds, d.label)
        d["y"] = y
        if name == "std":
            col, roles = "role_std", {"train": "train", "val": "val", "test_seen": "test_seen", "test_unseen": "test_unseen"}
        elif name in ("b0", "b1"):
            col, roles = f"b_f{name[1]}", {"train": "train", "val": "val", "seen": "seen", "unseen": "unseen"}
        if name != "dmc":
            for r, s in roles.items():
                x = d[d[col] == r]
                sets[s] = _sel(x.key, x.y)
        else:
            for s in ("train", "val", "test"):
                lab = _npz_n(RAW / "dermamnist_ce/dermamnist_corrected_224.npz", s)
                sets[{"test": "c_test"}.get(s, s)] = _sel([f"C:{s}:{i}" for i in range(len(lab))], lab)
        lab = _npz_n(RAW / "dermamnist_ce/dermamnist_extended_224.npz", "test")
        sets["e_test"] = _sel([f"E:test:{i}" for i in range(len(lab))], lab)
        pad = sorted(p.name for p in (REPO / "data/raw/pad_ufes20/images").rglob("*.png"))
        sets["ood"] = _sel(["pad:" + p for p in pad], [-1] * len(pad))
        isic = pd.read_csv(SPL / "isic2019.csv.gz")
        bcn = isic.image[isic.source == "BCN"]
        sets["ood_bcn"] = _sel(["isic:" + i for i in bcn], [-1] * len(bcn))
        nb = len(_npz_n(RAW / "medmnist/bloodmnist_224.npz", "test"))
        sets["ood_blood"] = _sel([f"blood:test:{i}" for i in range(nb)], [-1] * nb)
    elif ds == "isic2019":
        d = pd.read_csv(SPL / "isic2019.csv.gz")
        d["y"] = _lab(ds, d.label)
        d["k"] = "isic:" + d.image
        col = "role_std" if name == "std" else f"b_f{name[1]}"
        roles = ["train", "val", "test_seen", "test_unseen"] if name == "std" else ["train", "val", "seen", "unseen"]
        for r in roles:
            x = d[d[col] == r]
            sets[r] = _sel(x.k, x.y)
        x = d[d.role_std == "ood"]
        sets["ood"] = _sel(x.k, [-1] * len(x))
        pad = sorted(p.name for p in (REPO / "data/raw/pad_ufes20/images").rglob("*.png"))
        sets["ood_pad"] = _sel(["pad:" + p for p in pad], [-1] * len(pad))
    elif ds == "kermany":
        d = pd.read_csv(SPL / "kermany.csv.gz", low_memory=False)
        d["y"] = _lab(ds, d.label)
        d["k"] = d.version + ":" + d.key
        if name == "std":
            v = d[d.version == "v2"]
            for r in ("train", "val", "test_seen", "test_unseen"):
                x = v[v.role_std == r]
                sets[r] = _sel(x.k, x.y)
            x = d[d.role_std == "unseen_extra"]
            sets["unseen_extra"] = _sel(x.k, x.y)
            x = v[v.role_std == "ood"]
        else:
            v = d[(d.version == "v3") & (d.label != "CXR")]
            col = "role_v3std" if name == "v3std" else f"b_f{name[1]}"
            roles = ["train", "val", "test_unseen"] if name == "v3std" else ["train", "val", "seen", "unseen"]
            for r in roles:
                x = v[v[col] == r]
                sets[r] = _sel(x.k, x.y)
            x = v[v.label == "DRUSEN"]
        sets["ood"] = _sel(x.k, [-1] * len(x))
        x = d[d.label == "CXR"]
        sets["ood_cxr"] = _sel(x.k, [-1] * len(x))
    elif ds == "breakhis":
        d = pd.read_csv(SPL / "breakhis.csv.gz")
        d["y"] = _lab(ds, d.label)
        col = name.replace("std_r", "std_r").replace("gd_r", "gd_r")
        roles = ["train", "val", "seen", "unseen"] if name.startswith("std") else ["train", "val", "unseen"]
        for r in roles:
            x = d[d[col] == r]
            sets[r] = _sel(x.key, x.y)
        x = d[d.is_ood]
        sets["ood"] = _sel(x.key, [-1] * len(x))
        c = pd.read_csv(SPL / "crc_val.csv.gz")
        sets["ood_crc"] = _sel(c.key, [-1] * len(c))
    elif ds == "brain_cheng":
        d = pd.read_csv(SPL / "brain_cheng.csv.gz")
        d["y"] = _lab(ds, d.label)
        for r in ("train", "val", "test_seen", "test_unseen"):
            x = d[d.role_std == r]
            sets[r] = _sel(x.key, x.y)
        k = pd.read_csv(SPL / "kermany.csv.gz", low_memory=False)
        x = k[k.label == "CXR"]
        sets["ood"] = _sel("v3:" + x.key, [-1] * len(x))
    else:
        raise ValueError(ds)
    return sets


def arms_for(ds: str) -> list[str]:
    return {"dermamnist": ["std", "b0", "b1", "dmc"], "isic2019": ["std", "b0", "b1"],
            "kermany": ["std", "b0", "b1", "v3std"],
            "breakhis": [f"std_r{r}" for r in range(5)] + [f"gd_r{r}" for r in range(5)],
            "brain_cheng": ["std"]}[ds]


def grid(ds: str) -> list[tuple[str, int, str]]:
    """(arch, seed, arm) runs of a dataset, L2/L3: M_std arms 8 archs at seed 42 + 4 core at seed 43;
    group-disjoint / external / descriptive arms 4 core x seeds 42, 43."""
    out = []
    for a in arms_for(ds):
        std_like = a == "std" or a.startswith("std_r")
        for s in (42, 43):
            for arch in CORE + (ZOO if std_like and s == 42 else []):
                out.append((arch, s, a))
    return out
