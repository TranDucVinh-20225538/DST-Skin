#!/usr/bin/env python3
"""Medbench Phase 0 split audit and split construction
(decisions/precommit_group_leakage_medbench_2026-10-06.md, "Phase 0", L1, L3).

Reads only archive listings / metadata (images only where a check needs pixels: DermaMNIST mapping
verification, brain .mat PIDs). Writes outputs/reports/rigor_pack/medbench/phase0_split_audit.csv/.md
and splits/<dataset>.csv.gz. Group-fold / repeat / subsample RNG: default_rng(0).

Split CSV columns: key (archive member or image id), label (class name), group, role_std
(train / val / test_seen / test_unseen / unseen_extra / ood / ood2 / unused), fold (0/1 for ID groups, -1 else),
b_f0 / b_f1 (role in the group-disjoint arm trained on fold f: train / val / seen / unseen / ood / -),
plus dataset-specific columns.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/medbench"
OUT = REPO / "outputs/reports/rigor_pack/medbench"
ROWS: list[dict] = []


def rec(ds, check, value, expected=None, note=""):
    rel = None
    if expected is not None and isinstance(value, (int, float)) and expected != 0:
        rel = abs(value - expected) / abs(expected)
    status = "" if rel is None else ("S3-PAUSE" if rel > 0.25 else "ok")
    ROWS.append({"dataset": ds, "check": check, "value": value, "expected": expected,
                 "rel_diff": None if rel is None else round(rel, 4), "status": status, "note": note})
    print(f"[{ds}] {check}: {value} (expected {expected}) {status} {note}", flush=True)


# ---------------------------------------------------------------- shared split helpers
def assign_folds(groups: np.ndarray, classes: np.ndarray, rng) -> dict:
    """Groups shuffled within class stratum (group class = majority class), greedily assigned to the
    fold with fewer images so far (tie -> 0)."""
    df = pd.DataFrame({"g": groups, "c": classes})
    gc = df.groupby("g").c.agg(lambda s: s.value_counts().index[0])
    gn = df.groupby("g").size()
    fold_of = {}
    for c in sorted(gc.unique()):
        gs = np.array(sorted(gc.index[gc == c]))
        rng.shuffle(gs)
        n = [0, 0]
        for g in gs:
            f = 0 if n[0] <= n[1] else 1
            fold_of[g] = f
            n[f] += int(gn[g])
    return fold_of


def b_roles(df: pd.DataFrame, id_mask: np.ndarray, rng) -> None:
    """Group-disjoint arm per fold f: fold-f images -> train, except 15% of the images of fold-f groups
    with >= 2 images held out as seen (>= 1 image of each group stays in train); then 10% of the
    remaining train images -> val; fold-(1-f) images -> unseen."""
    for f in (0, 1):
        role = np.array(["-"] * len(df), dtype=object)
        role[id_mask & (df.fold.to_numpy() == 1 - f)] = "unseen"
        inf = np.flatnonzero(id_mask & (df.fold.to_numpy() == f))
        role[inf] = "train"
        sub = df.iloc[inf]
        counts = sub.group.value_counts()
        multi = sub[sub.group.isin(counts.index[counts >= 2])]
        cand = []
        for g, ix in multi.groupby("group").groups.items():
            ix = np.array(sorted(ix))
            keep = rng.choice(ix)
            cand.extend([i for i in ix if i != keep])
        n_seen = int(round(0.15 * len(multi)))
        cand = np.array(sorted(cand))
        seen = rng.choice(cand, size=min(n_seen, len(cand)), replace=False) if len(cand) else np.array([], int)
        role[df.index.get_indexer(seen)] = "seen"
        tr = np.flatnonzero(role == "train")
        val = rng.choice(tr, size=int(round(0.10 * len(tr))), replace=False)
        role[val] = "val"
        df[f"b_f{f}"] = role


def std_val(df: pd.DataFrame, rng) -> None:
    tr = np.flatnonzero(df.role_std.to_numpy() == "train")
    val = rng.choice(tr, size=int(round(0.10 * len(tr))), replace=False)
    df.loc[df.index[val], "role_std"] = "val"


def leak_share(df, eval_roles=("test_seen", "test_unseen")) -> float:
    e = df[df.role_std.isin(eval_roles)]
    return float((e.role_std == "test_seen").mean()) if len(e) else float("nan")


def arm_sizes(ds, df, id_mask):
    for f in (0, 1):
        u = df[id_mask & (df[f"b_f{f}"] == "unseen")]
        s = df[id_mask & (df[f"b_f{f}"] == "seen")]
        tr = df[id_mask & (df[f"b_f{f}"] == "train")]
        rec(ds, f"B fold{f}: train / val / seen / unseen images", f"{len(tr)} / {(df[f'b_f{f}'] == 'val').sum()} / {len(s)} / {len(u)}")
        rec(ds, f"B fold{f}: unseen groups", int(u.group.nunique()),
            note="underpowered" if (len(u) < 200 or u.group.nunique() < 20) else "")


def save(name, df):
    p = OUT / "splits" / f"{name}.csv.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def epochs(n):
    return int(min(50, max(10, int(np.ceil(300000 / n)))))


# ---------------------------------------------------------------- DermaMNIST
MEDMNIST_DERMA = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


def dermamnist(rng):
    ds = "dermamnist"
    D = RAW / "dermamnist_ce"
    z = np.load(RAW / "medmnist/dermamnist.npz")
    info = pd.read_csv(D / "dermamnist_split_info.csv")
    ham = pd.read_csv(D / "HAM10000_metadata.csv")
    for s, n in (("train", 7007), ("val", 1003), ("test", 2005)):
        rec(ds, f"{s} images (npz)", int(len(z[f"{s}_labels"])), n)
        rec(ds, f"{s} rows in mapping CSV", int((info.split == s).sum()), int(len(z[f"{s}_labels"])))
    df = info.merge(ham[["image_id", "lesion_id", "dx"]], on="image_id", how="left")
    lab = np.concatenate([z[f"{s}_labels"].ravel() for s in ("train", "val", "test")])
    df = df.sort_values(["split", "index"], key=lambda c: c.map({"train": 0, "val": 1, "test": 2}) if c.name == "split" else c)
    df = df.reset_index(drop=True)
    df["npz_label"] = [MEDMNIST_DERMA[int(x)] for x in lab]
    rec(ds, "images with lesion_id (fraction)", float(df.lesion_id.notna().mean()), 1.0)
    rec(ds, "mapping label agreement (npz label == HAM dx)", float((df.npz_label == df.dx).mean()), 1.0)
    # pixel verification on a random sample (28 px npz vs ISIC 2018 Task 3 training JPEG)
    from PIL import Image
    img_dir = REPO / "data/raw/isic2018/ISIC2018_Task3_Training_Input"
    imgs = np.concatenate([z[f"{s}_images"] for s in ("train", "val", "test")])
    samp = rng.choice(len(df), size=300, replace=False)
    rho = []
    for i in samp:
        a = imgs[i].astype(np.float64).ravel()
        im = Image.open(img_dir / f"{df.image_id[i]}.jpg").convert("RGB")
        w, h = im.size
        s = min(w, h)
        sq = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))  # MedMNIST centre-crops
        best = -1.0
        for src in (im, sq):
            for rs in (Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.BOX):
                b = np.asarray(src.resize((28, 28), rs), dtype=np.float64).ravel()
                best = max(best, np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
        rho.append(best)
    rho = np.array(rho)
    rec(ds, "pixel check: fraction of 300 sampled images with rho >= 0.98", float((rho >= 0.98).mean()), 1.0,
        f"median rho {np.median(rho):.4f}, min {rho.min():.4f}")
    les = {s: set(df.lesion_id[df.split == s]) for s in ("train", "val", "test")}
    for a, b, exp_img, exp_les in (("train", "test", 886, 641), ("train", "val", None, None), ("val", "test", None, None)):
        m = (df.split == b) & df.lesion_id.isin(les[a])
        rec(ds, f"{b} images whose lesion is in {a}", int(m.sum()), exp_img)
        rec(ds, f"{b} lesions shared with {a}", int(len(les[a] & les[b])), exp_les)
    # DermaMNIST-C / -E disjointness
    c = pd.read_csv(D / "DermaMNIST-C.csv")
    lc = {s: set(c.lesion_id[c.split == s]) for s in c.split.unique()}
    rec(ds, "DermaMNIST-C lesions shared across splits", int(sum(len(lc[a] & lc[b]) for a in lc for b in lc if a < b)), 0)
    e = pd.read_csv(D / "DermaMNIST-E.csv")
    e = e.merge(ham[["image_id", "lesion_id"]].rename(columns={"image_id": "image"}), on="image", how="left")
    le = {s: set(e.lesion_id[(e.split == s) & e.lesion_id.notna()]) for s in e.split.unique()}
    rec(ds, "DermaMNIST-E lesions shared across splits (HAM images)", int(sum(len(le[a] & le[b]) for a in le for b in le if a < b)), 0)
    rec(ds, "DermaMNIST-E test images outside HAM (ISIC 2018 test, no lesion id)",
        int(((e.split == "test") & e.lesion_id.isna()).sum()), note="disjoint by source")
    v = sorted(p.stem for p in (REPO / "data/raw/isic2018/ISIC2018_Task3_Validation_Input").glob("*.jpg"))
    rec(ds, "repo ISIC 2018 val images with a public lesion id", int(ham.image_id.isin(v).sum()),
        note=f"of {len(v)}; val lesion ids not public -> train/val lesion overlap not determinable")

    # splits: M_std = official; groups = lesion
    df["key"] = df.split + ":" + df["index"].astype(str)
    df["label"] = df.npz_label
    df["group"] = df.lesion_id
    df["role_std"] = df.split.map({"train": "train", "val": "val", "test": "test"})
    t = df.role_std == "test"
    df.loc[t & df.group.isin(les["train"]), "role_std"] = "test_seen"
    df.loc[t & ~df.group.isin(les["train"]), "role_std"] = "test_unseen"
    s1 = leak_share(df)
    rec(ds, "S1: standard test images sharing a lesion with train (fraction)", s1, note="clean" if s1 < 0.05 else "leaky")
    fold_of = assign_folds(df.group.to_numpy(), df.label.to_numpy(), rng)
    df["fold"] = df.group.map(fold_of)
    b_roles(df, np.ones(len(df), bool), rng)
    arm_sizes(ds, df, np.ones(len(df), bool))
    rec(ds, "M_std train images / epochs", f"{(df.split == 'train').sum()} / {epochs((df.split == 'train').sum())}")
    tu = df[df.role_std == "test_unseen"]
    rec(ds, "A: test_seen / test_unseen images", f"{(df.role_std == 'test_seen').sum()} / {len(tu)}")
    rec(ds, "A: test_unseen lesions", int(tu.group.nunique()),
        note="underpowered" if (len(tu) < 200 or tu.group.nunique() < 20) else "")
    save(ds, df[["key", "image_id", "label", "group", "split", "role_std", "fold", "b_f0", "b_f1"]])


# ---------------------------------------------------------------- ISIC 2019
ISIC_ID = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]
ISIC_OOD = ["DF", "VASC"]


def isic2019(rng):
    ds = "isic2019"
    gt = pd.read_csv(RAW / "isic2019/ISIC_2019_Training_GroundTruth.csv")
    md = pd.read_csv(RAW / "isic2019/ISIC_2019_Training_Metadata.csv")
    cls = gt.columns[1:]
    gt["label"] = cls[gt[cls].to_numpy().argmax(1)]
    df = gt[["image", "label"]].merge(md[["image", "lesion_id"]], on="image", how="left")
    rec(ds, "images", int(len(df)), 25331)
    rec(ds, "images with lesion_id", int(df.lesion_id.notna().sum()), 23247)
    cov = float(df.lesion_id.notna().mean())
    rec(ds, "S2: lesion_id coverage", cov, note="STOP (<0.85)" if cov < 0.85 else "ok")
    rec(ds, "distinct lesions", int(df.lesion_id.nunique()), 11847)
    src = np.where(df.lesion_id.isna(), "no_lesion_id", df.lesion_id.astype(str).str.extract(r"^([A-Za-z]+)")[0])
    df["source"] = src
    for k, v in df.source.value_counts().items():
        rec(ds, f"composition: lesion prefix {k}", int(v))
    zl = zipfile.ZipFile(RAW / "isic2019/ISIC_2019_Training_Input.zip").namelist()
    stems = {Path(n).stem for n in zl if n.endswith(".jpg")}
    rec(ds, "images present in zip", int(df.image.isin(stems).sum()), int(len(df)))
    df["group"] = np.where(df.lesion_id.isna(), "IMG:" + df.image, df.lesion_id)
    idm = df.label.isin(ISIC_ID).to_numpy()
    ood_les = set(df.lesion_id[df.label.isin(ISIC_OOD) & df.lesion_id.notna()])
    id_les = set(df.lesion_id[idm & df.lesion_id.notna()])
    clash = df.label.isin(ISIC_OOD) & df.lesion_id.isin(id_les)
    rec(ds, "DF/VASC images whose lesion also has an ID-class image (removed from OOD)", int(clash.sum()), 0)
    df["role_std"] = "unused"
    df.loc[df.label.isin(ISIC_OOD) & ~clash, "role_std"] = "ood"
    # M_std: class-stratified random 80/20 over ID images
    for c in ISIC_ID:
        ix = np.flatnonzero((df.label == c).to_numpy())
        rng.shuffle(ix)
        nt = int(round(0.2 * len(ix)))
        df.loc[df.index[ix[:nt]], "role_std"] = "test"
        df.loc[df.index[ix[nt:]], "role_std"] = "train"
    trg = set(df.group[df.role_std == "train"])
    t = df.role_std == "test"
    df.loc[t & df.group.isin(trg), "role_std"] = "test_seen"
    df.loc[t & ~df.group.isin(trg), "role_std"] = "test_unseen"
    s1 = leak_share(df)
    rec(ds, "S1/expected: test images whose lesion is in train (fraction)", s1, 0.60, "clean" if s1 < 0.05 else "leaky")
    n_tr = int((df.role_std == "train").sum())
    std_val(df, rng)
    rec(ds, "M_std train images (before val) / epochs", f"{n_tr} / {epochs(n_tr)}")
    tu = df[df.role_std == "test_unseen"]
    rec(ds, "A: test_seen / test_unseen images", f"{(df.role_std == 'test_seen').sum()} / {len(tu)}")
    rec(ds, "A: test_unseen groups", int(tu.group.nunique()))
    rec(ds, "OOD images (DF + VASC)", int((df.role_std == "ood").sum()))
    fold_of = assign_folds(df.group[idm].to_numpy(), df.label[idm].to_numpy(), rng)
    df["fold"] = df.group.map(fold_of).fillna(-1).astype(int)
    b_roles(df, idm, rng)
    for f in (0, 1):
        df.loc[df.role_std == "ood", f"b_f{f}"] = "ood"
    arm_sizes(ds, df, idm)
    # sensitivity: drop images without lesion_id
    sens = df[idm & df.lesion_id.notna()]
    rec(ds, "sensitivity (lesion_id only): ID images", int(len(sens)))
    save(ds, df[["image", "label", "lesion_id", "group", "source", "role_std", "fold", "b_f0", "b_f1"]])


# ---------------------------------------------------------------- Kermany
KER_ID = ["CNV", "DME", "NORMAL"]
PID = re.compile(r"^(CNV|DME|DRUSEN|NORMAL)-(\d+)-(\d+)\.jpe?g$")


def tar_members(p: Path) -> list[str]:
    out = subprocess.run(["tar", "tzf", str(p)], capture_output=True, text=True, check=True).stdout
    return [m for m in out.splitlines() if not m.endswith("/")]


def kermany(rng):
    ds = "kermany"
    import hashlib
    v2 = [m for m in tar_members(RAW / "kermany_v2/OCT2017.tar.gz") if m.lower().endswith((".jpeg", ".jpg"))]
    v3 = [m for m in zipfile.ZipFile(RAW / "kermany_v3/ZhangLabData.zip").namelist()
          if m.startswith("CellData/OCT/") and m.lower().endswith((".jpeg", ".jpg"))]
    rows = []
    for ver, mem in (("v2", v2), ("v3", v3)):
        for m in mem:
            parts = m.split("/")
            split, cl, fn = parts[-3], parts[-2], parts[-1]
            mm = PID.match(fn)
            rows.append({"version": ver, "key": m, "split": split, "label": cl, "fname": fn,
                         "pid": mm.group(2) if mm else None})
    df = pd.DataFrame(rows)
    for ver in ("v2", "v3"):
        d = df[df.version == ver]
        sha = hashlib.sha256("\n".join(sorted(d.fname + "|" + d.split)).encode()).hexdigest()[:16]
        rec(ds, f"{ver} fingerprint: counts by split", d.split.value_counts().to_dict().__repr__(), note=f"sha256(sorted names)[:16]={sha}")
        rec(ds, f"{ver} has val/ folder", bool((d.split == "val").any()))
        pr = float(d.pid.notna().mean())
        rec(ds, f"S2: {ver} patient-id parse rate", pr, note="STOP (<0.90)" if pr < 0.9 else "ok")
    v2d, v3d = df[df.version == "v2"], df[df.version == "v3"]
    rec(ds, "v2 images (train / val / test)", f"{(v2d.split == 'train').sum()} / {(v2d.split == 'val').sum()} / {(v2d.split == 'test').sum()}",
        note="expected 83484 / 32 / 968")
    p2tr = set(v2d.pid[v2d.split == "train"])
    p3tr = set(v3d.pid[v3d.split == "train"])
    t2, t3 = v2d[v2d.split == "test"], v3d[v3d.split == "test"]
    rec(ds, "v2 test images whose patient is in v2 train (fraction)", float(t2.pid.isin(p2tr).mean()), 0.92)
    rec(ds, "v3 test images whose patient is in v3 train (fraction)", float(t3.pid.isin(p3tr).mean()), note="expected 0")
    rec(ds, "v3 test images whose patient is in v2 train (fraction)", float(t3.pid.isin(p2tr).mean()))
    rec(ds, "v3 filenames also present in v2 (fraction)", float(v3d.fname.isin(set(v2d.fname)).mean()))
    rec(ds, "v3 test filenames present in v2 (any split) (fraction)", float(t3.fname.isin(set(v2d.fname)).mean()))
    rec(ds, "v3 test filenames present in v2 test (fraction)", float(t3.fname.isin(set(t2.fname)).mean()))
    dr_p = set(df.pid[df.label == "DRUSEN"])
    id_p = set(df.pid[df.label.isin(KER_ID)])
    rec(ds, "DRUSEN patients also with ID-class images (count, both versions)", int(len(dr_p & id_p)), note="descriptive")

    # M_std (v2) + (A)
    v2d = v2d.copy()
    v2d["group"] = v2d.pid
    v2d["role_std"] = "unused"
    idm2 = v2d.label.isin(KER_ID)
    v2d.loc[idm2 & v2d.split.isin(["train", "val"]), "role_std"] = "train"
    t = idm2 & (v2d.split == "test")
    v2d.loc[t & v2d.pid.isin(p2tr), "role_std"] = "test_seen"
    v2d.loc[t & ~v2d.pid.isin(p2tr), "role_std"] = "test_unseen"
    v2d.loc[v2d.label == "DRUSEN", "role_std"] = "ood"
    s1 = leak_share(v2d)
    rec(ds, "S1: v2 standard test (ID classes) sharing a patient with train", s1, note="clean" if s1 < 0.05 else "leaky")
    n_tr = int((v2d.role_std == "train").sum())
    std_val(v2d, rng)
    rec(ds, "M_std(v2) train images (ID, before val) / epochs", f"{n_tr} / {epochs(n_tr)}")
    v3d = v3d.copy()
    v3d["group"] = v3d.pid
    extra = v3d.label.isin(KER_ID) & (v3d.split == "test") & ~v3d.pid.isin(p2tr) & ~v3d.fname.isin(set(t2.fname))
    v3d["role_std"] = np.where(extra, "unseen_extra", "unused")
    tu = pd.concat([v2d[v2d.role_std == "test_unseen"], v3d[v3d.role_std == "unseen_extra"]])
    rec(ds, "A: test_seen / test_unseen(v2) / unseen_extra(v3 test) images",
        f"{(v2d.role_std == 'test_seen').sum()} / {(v2d.role_std == 'test_unseen').sum()} / {int(extra.sum())}")
    rec(ds, "A: ID_unseen patients (v2 test unseen + v3 extra)", int(tu.group.nunique()),
        note="underpowered" if (len(tu) < 200 or tu.group.nunique() < 20) else "")
    rec(ds, "A OOD: v2 DRUSEN images (all splits)", int((v2d.role_std == "ood").sum()))
    # (B) rebuild on v3 train
    idm3 = (v3d.label.isin(KER_ID) & (v3d.split == "train")).to_numpy()
    fold_of = assign_folds(v3d.group[idm3].to_numpy(), v3d.label[idm3].to_numpy(), rng)
    v3d["fold"] = np.where(idm3, v3d.group.map(fold_of), -1)
    v3d = v3d.reset_index(drop=True)
    b_roles(v3d, idm3, rng)
    for f in (0, 1):
        v3d.loc[v3d.label == "DRUSEN", f"b_f{f}"] = "ood"
    arm_sizes(ds, v3d, idm3)
    # descriptive arm M_std(v3): official v3 split
    v3d["role_v3std"] = np.where(v3d.label == "DRUSEN", "ood",
                                 np.where(v3d.label.isin(KER_ID), np.where(v3d.split == "train", "train", "test_unseen"), "unused"))
    tr3 = np.flatnonzero(v3d.role_v3std.to_numpy() == "train")
    v3d.loc[v3d.index[rng.choice(tr3, size=int(round(0.1 * len(tr3))), replace=False)], "role_v3std"] = "val"
    v2d["fold"] = -1
    out = pd.concat([v2d, v3d], ignore_index=True)
    for c in ("b_f0", "b_f1", "role_v3std"):
        out[c] = out[c].fillna("-")
    # far OOD: pediatric CXR test (v3)
    cxr = [m for m in zipfile.ZipFile(RAW / "kermany_v3/ZhangLabData.zip").namelist()
           if m.startswith("CellData/chest_xray/test/") and m.lower().endswith((".jpeg", ".jpg"))]
    rec(ds, "far OOD: pediatric CXR test images (v3)", len(cxr), 624)
    out = pd.concat([out, pd.DataFrame({"version": "v3", "key": cxr, "split": "test", "label": "CXR",
                                        "role_std": "ood2", "fold": -1, "b_f0": "ood2", "b_f1": "ood2", "role_v3std": "ood2"})],
                    ignore_index=True)
    save(ds, out[["version", "key", "split", "label", "pid", "group", "role_std", "fold", "b_f0", "b_f1", "role_v3std"]])


# ---------------------------------------------------------------- BreakHis
BH = re.compile(r"SOB_([BM])_([A-Z]+)-(\d+)-([0-9A-Z]+)-(\d+)-(\d+)\.png$")
BH_ID = ["A", "F", "TA", "DC", "LC", "MC"]
BH_OOD = ["PT", "PC"]


def breakhis(rng):
    ds = "breakhis"
    mem = [m for m in tar_members(RAW / "breakhis/BreaKHis_v1.tar.gz") if m.endswith(".png")]
    rows, bad = [], 0
    for m in mem:
        mm = BH.search(m)
        if not mm:
            bad += 1
            continue
        rows.append({"key": m, "label": mm.group(2), "patient": f"{mm.group(3)}-{mm.group(4)}", "mag": int(mm.group(5))})
    df = pd.DataFrame(rows)
    rec(ds, "images", int(len(df)), 7909)
    rec(ds, "S2: filename parse rate", 1 - bad / max(len(mem), 1), note="STOP (<0.90)" if bad / max(len(mem), 1) > 0.1 else "ok")
    rec(ds, "patients", int(df.patient.nunique()), 82)
    ipp = df.groupby("patient").size()
    rec(ds, "images per patient (min / median / max)", f"{ipp.min()} / {int(ipp.median())} / {ipp.max()}")
    ns = df.groupby("patient").label.nunique()
    rec(ds, "patients in > 1 subtype", int((ns > 1).sum()), note=", ".join(ns.index[ns > 1]))
    df["group"] = df.patient
    id_pat = set(df.patient[df.label.isin(BH_ID)])
    clash = df.label.isin(BH_OOD) & df.patient.isin(id_pat)
    rec(ds, "OOD-subtype images of patients who also have ID images (removed from OOD)", int(clash.sum()))
    df["is_ood"] = df.label.isin(BH_OOD) & ~clash
    idm = df.label.isin(BH_ID).to_numpy()
    rec(ds, "ID images / OOD images", f"{idm.sum()} / {int(df.is_ood.sum())}")
    # image-level split leak share (what a random image split gives)
    ix = np.flatnonzero(idm)
    perm = rng.permutation(ix)
    te = perm[: int(round(0.2 * len(ix)))]
    trp = set(df.patient.iloc[perm[int(round(0.2 * len(ix))):]])
    s1 = float(df.patient.iloc[te].isin(trp).mean())
    rec(ds, "S1: random image 80/20 split, test images whose patient is in train", s1, note="clean" if s1 < 0.05 else "leaky")
    # 5 repeats = stratified 5-fold patient partition (P_out_r = fold r), seed 0
    pc = df[idm].groupby("patient").label.agg(lambda s: s.value_counts().index[0])
    rep_of = {}
    for c in sorted(pc.unique()):
        ps = np.array(sorted(pc.index[pc == c]))
        rng.shuffle(ps)
        for j, p in enumerate(ps):
            rep_of[p] = j % 5
    df["p_out_repeat"] = df.patient.map(rep_of).fillna(-1).astype(int)
    for r in range(5):
        role = np.array(["-"] * len(df), dtype=object)
        role[df.is_ood.to_numpy()] = "ood"
        pout = idm & (df.p_out_repeat.to_numpy() == r)
        role[pout] = "unseen"
        pin = np.flatnonzero(idm & (df.p_out_repeat.to_numpy() != r))
        pin = rng.permutation(pin)
        nseen = int(round(0.2 * len(pin)))
        role[pin[:nseen]] = "seen"
        role[pin[nseen:]] = "train"
        trn = pin[nseen:]
        role[rng.choice(trn, size=int(round(0.1 * len(trn))), replace=False)] = "val"
        df[f"std_r{r}"] = role
        # M_gd(r): all P_in images, 10% val
        g = np.array(["-"] * len(df), dtype=object)
        g[df.is_ood.to_numpy()] = "ood"
        g[pout] = "unseen"
        g[pin] = "train"
        g[rng.choice(pin, size=int(round(0.1 * len(pin))), replace=False)] = "val"
        df[f"gd_r{r}"] = g
        rec(ds, f"repeat {r}: P_out patients / images; M_std train (before val) / seen",
            f"{df.patient[pout].nunique()} / {int(pout.sum())}; {len(trn)} / {nseen}",
            note="per-repeat cell < 20 groups (underpowered); pooled over repeats" if df.patient[pout].nunique() < 20 else "")
    n_tr = int(np.mean([(df[f"std_r{r}"].isin(["train", "val"])).sum() for r in range(5)]))
    rec(ds, "M_std(r) train images (mean, before val) / epochs", f"{n_tr} / {epochs(n_tr)}")
    rec(ds, "pooled ID_unseen over 5 repeats: patients / images", f"{df.patient[idm].nunique()} / {int(idm.sum())}")
    # CRC far OOD
    crc = [m for m in zipfile.ZipFile(RAW / "crc_val/CRC-VAL-HE-7K.zip").namelist() if m.endswith(".tif")]
    rec(ds, "far OOD: CRC-VAL-HE-7K images", len(crc), 7180)
    save(ds, df[["key", "label", "patient", "group", "mag", "is_ood", "p_out_repeat"]
                + [f"std_r{r}" for r in range(5)] + [f"gd_r{r}" for r in range(5)]])
    save("crc_val", pd.DataFrame({"key": crc, "label": [Path(m).parent.name for m in crc]}))


# ---------------------------------------------------------------- Brain (Cheng)
def brain(rng, tmp: Path):
    ds = "brain_cheng"
    import h5py
    import scipy.io as sio
    rows = []
    for z in sorted((RAW / "brain_cheng").glob("figshare_*.zip")):
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            if not n.endswith(".mat"):
                continue
            with h5py.File(io.BytesIO(zf.read(n)), "r") as h:
                cj = h["cjdata"]
                pid = "".join(chr(int(c)) for c in np.asarray(cj["PID"]).ravel())
                rows.append({"key": f"{z.name}:{n}", "idx": int(Path(n).stem), "label": int(np.asarray(cj["label"]).ravel()[0]),
                             "pid": pid.strip("\x00 ")})
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    rec(ds, "images", int(len(df)), 3064)
    rec(ds, "patients (PID)", int(df.pid.nunique()), 233)
    rec(ds, "S2: PID present", float((df.pid != "").mean()))
    try:
        cv = sio.loadmat(RAW / "brain_cheng/cvind.mat")
        key = [k for k in cv if not k.startswith("__")][0]
        cvi = np.asarray(cv[key]).ravel(order="F").astype(int)
    except NotImplementedError:  # v7.3 file
        with h5py.File(RAW / "brain_cheng/cvind.mat", "r") as h:
            key = list(h.keys())[0]
            cvi = np.asarray(h[key]).T.ravel(order="F").astype(int)
    rec(ds, "cvind length / folds", f"{len(cvi)} / {sorted(set(cvi.tolist()))}", note=f"variable {key}")
    df["cvind"] = cvi[: len(df)]
    per = df.groupby("pid").cvind.nunique()
    rec(ds, "patients spanning > 1 cvind fold", int((per > 1).sum()), 0)
    df["group"] = df.pid
    df["label"] = df.label.map({1: "meningioma", 2: "glioma", 3: "pituitary"})
    df["role_std"] = "train"
    for c in sorted(df.label.unique()):
        ix = np.flatnonzero((df.label == c).to_numpy())
        rng.shuffle(ix)
        df.loc[df.index[ix[: int(round(0.2 * len(ix)))]], "role_std"] = "test"
    trg = set(df.group[df.role_std == "train"])
    t = df.role_std == "test"
    df.loc[t & df.group.isin(trg), "role_std"] = "test_seen"
    df.loc[t & ~df.group.isin(trg), "role_std"] = "test_unseen"
    s1 = leak_share(df)
    rec(ds, "S1: image 80/20 test images sharing a patient with train", s1, note="clean" if s1 < 0.05 else "leaky")
    n_tr = int((df.role_std == "train").sum())
    std_val(df, rng)
    rec(ds, "M_std train images (before val) / epochs", f"{n_tr} / {epochs(n_tr)}")
    save(ds, df[["key", "idx", "label", "pid", "group", "cvind", "role_std"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dermamnist", "isic2019", "kermany", "breakhis", "brain_cheng"])
    ap.add_argument("--tmp", default="/tmp")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"dermamnist": dermamnist, "isic2019": isic2019, "kermany": kermany, "breakhis": breakhis}
    for ds in args.datasets:
        rng = np.random.default_rng(0)
        try:
            if ds == "brain_cheng":
                brain(rng, Path(args.tmp))
            else:
                fns[ds](rng)
        except Exception as e:  # recorded, the run continues with the next dataset
            import traceback
            traceback.print_exc()
            rec(ds, "ERROR", repr(e)[:300])
    a = pd.DataFrame(ROWS)
    name = "phase0_split_audit" if len(args.datasets) == 5 else "phase0_split_audit_" + "_".join(args.datasets)
    a.to_csv(OUT / f"{name}.csv", index=False)
    md = ["# Medbench Phase 0 split audit", "",
          "Precommit `decisions/precommit_group_leakage_medbench_2026-10-06.md` (0fa52af). RNG default_rng(0) per dataset.",
          "status: ok = within 25% of the expected value; S3-PAUSE = > 25% off (look for a mapping / parse bug).", ""]
    for ds, g in a.groupby("dataset", sort=False):
        md += [f"## {ds}", "", "| check | value | expected | status | note |", "|---|---|---|---|---|"]
        for _, r in g.iterrows():
            v = f"{r.value:.4f}" if isinstance(r.value, float) else str(r.value)
            md.append(f"| {r.check} | {v} | {'' if pd.isna(r.expected) else r.expected} | {r.status} | {r.note} |")
        md.append("")
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
