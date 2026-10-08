#!/usr/bin/env python3
"""R3 item 8 / P2-d Phase 0 (results/r3/8/p2d/PRECOMMIT.json): audit, 256 px staging and set construction.

kvasir: reads data/raw/kvasir_capsule/kvasir-capsule-labeled-images.zip (one tar.gz per class folder; members are JPEG). Every image:
  video / frame from the archive member name, parsed by str.rsplit and by the strict regex (third check against the
  split files' filenames), RGB size, pixel hash; staged RGB 256 x 256 bicubic into data/staged/p2d_kvasir_256/{class}.tar.
  Audit: three-way agreement, official-fold overlap recounted with archive video IDs, duplicate pixels across videos.
  Sets: labels = the 11 split-file labels (the 3 classes the authors dropped for their baseline are counted, not used;
  8 two-label files dropped). Per OOD candidate (Angiectasia, Erosion): OOD-class videos removed; 20% unseen videos
  (stratified on majority class), 10% ckpt; K_lit = frame-level random 70 / 15 / 15 stratified on class; K_seg =
  contiguous same-label runs 70 / 15 / 15 by frames; A-fit folds (medbench rule) over each variant's fit videos;
  gap25 = K_seg fit frame within +/- 25 frames of a K_seg seen frame of the same video. RNG sequence identical to
  scripts/r3/item8_p2d_design_check.py (asserted on the fit counts).
brain: reads the 4 .mat zips (MATLAB v7.3 / HDF5): PID, label, image int16 -> per-slice min-max uint8 (README) -> grey
  256 x 256 bicubic into data/staged/p2d_brain_256/brain.tar. Audit: counts vs README, slices per patient, patients per
  class, duplicate pixels, fields (no slice index). Sets per held-out class (meningioma, pituitary, glioma): 20% unseen
  patients (stratified on class), 10% ckpt, record-wise random 60 / 20 / 20 stratified on class, A-fit folds.
Writes results/r3/8/p2d/{kvasir,brain}_{images,sets}.csv.gz and {kvasir,brain}_phase0.json
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
sys.path.insert(0, str(REPO / "scripts/r3"))
from medbench_phase0 import assign_folds  # noqa: E402
import item8_p2d_design_check as dc  # noqa: E402

OUT = REPO / "results/r3/8/p2d"
RX = re.compile(r"^([0-9a-f]{16})_([0-9]+)\.(?:jpg|png)$")
FOLDER2LABEL = {"normal_clean_mucosa": "Normal", "ileocecal_valve": "Ileo-cecal valve", "reduced_mucosal_view": "Reduced Mucosal View",
                "pylorus": "Pylorus", "angiectasia": "Angiectasia", "ulcer": "Ulcer", "foreign_body": "Foreign Bodies",
                "lymphangiectasia": "Lymphangiectasia", "erosion": "Erosion", "blood_fresh": "Blood", "erythema": "Erythematous"}
DROPPED = ("ampulla_of_vater", "blood_hematin", "polyp")
BRAIN = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def stage_png(im, size, mode):
    from PIL import Image
    b = io.BytesIO()
    im.convert(mode).resize((size, size), Image.BICUBIC).save(b, format="PNG")
    return b


def add(tar, name, b):
    ti = tarfile.TarInfo(name)
    ti.size = b.tell()
    b.seek(0)
    tar.addfile(ti, b)


def kvasir_read():
    from PIL import Image
    stage = REPO / "data/staged/p2d_kvasir_256"
    stage.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(REPO / "data/raw/kvasir_capsule/kvasir-capsule-labeled-images.zip")
    rows = []
    for zn in sorted(n for n in z.namelist() if n.endswith(".tar.gz")):
        folder = Path(zn).name[: -len(".tar.gz")]
        part = stage / f"{folder}.tar.part"
        with z.open(zn) as fz, tarfile.open(fileobj=fz, mode="r|gz") as t, tarfile.open(part, "w") as to:
            for m in t:
                if not (m.isfile() and m.name.lower().endswith((".jpg", ".png"))):
                    continue
                name = Path(m.name).name
                im = Image.open(io.BytesIO(t.extractfile(m).read()))
                im.load()
                mx = RX.match(name)
                va, fa = name.rsplit("_", 1)
                fa = fa.rsplit(".", 1)[0]
                rows.append({"file": name[:-4], "folder": folder, "video_a": va, "frame_a": int(fa),
                             "video_b": mx.group(1) if mx else None, "frame_b": int(mx.group(2)) if mx else None,
                             "w": im.size[0], "h": im.size[1], "mode": im.mode, "format": im.format,
                             "pix": hashlib.sha1(np.asarray(im.convert("RGB")).tobytes()).hexdigest()})
                add(to, name[:-4] + ".png", stage_png(im, 256, "RGB"))
        part.rename(stage / f"{folder}.tar")
        print(folder, sum(r["folder"] == folder for r in rows), flush=True)
    return pd.DataFrame(rows)


def kvasir_build(u, ood_class):
    """Same RNG sequence as dc.sets + dc.segment_roles, keeping unseen / ckpt rows."""
    vs_ood = set(u.loc[u.label == ood_class, "video"])
    x = u[~u.video.isin(vs_ood)].copy()
    major = x.groupby("video").label.agg(lambda s: s.value_counts().index[0])
    rng = np.random.default_rng(0)
    unseen = set()
    for _, g in major.groupby(major):
        g = np.array(sorted(g.index))
        unseen |= set(g[rng.permutation(len(g))[: int(round(0.2 * len(g)))]])
    vids = np.array(sorted(x.video.unique()))
    rest = np.array([v for v in vids if v not in unseen])
    ckpt = set(rest[rng.permutation(len(rest))[: int(round(0.1 * len(rest)))]])
    x["role_lit"] = np.where(x.video.isin(unseen), "unseen", np.where(x.video.isin(ckpt), "ckpt", ""))
    y = x[x.role_lit == ""].copy()
    role = np.empty(len(y), dtype=object)
    for _, idx in y.groupby("label").indices.items():
        p = rng.permutation(idx)
        n_tr, n_va = int(round(0.70 * len(p))), int(round(0.15 * len(p)))
        role[p[:n_tr]], role[p[n_tr:n_tr + n_va]], role[p[n_tr + n_va:]] = "fit", "val", "seen"
    y["role_lit"] = role
    z = dc.segment_roles(y.drop(columns="role_lit").assign(role="x"))
    y = y.join(z[["seg", "role"]].rename(columns={"role": "role_seg"}))
    y["role_seg"] = y.role_seg.replace({"test": "seen"})
    x = x.join(y[["role_lit", "seg", "role_seg"]], rsuffix="_y")
    x["role_lit"] = np.where(x.role_lit == "", x.role_lit_y, x.role_lit)
    x["role_seg"] = np.where(x.role_seg.isna(), x.role_lit, x.role_seg)
    x = x.drop(columns="role_lit_y")
    for var in ("lit", "seg"):
        f = x[f"role_{var}"] == "fit"
        fo = assign_folds(x.loc[f, "video"].to_numpy(), x.loc[f, "label"].to_numpy(), np.random.default_rng(0))
        x[f"afit_fold_{var}"] = x.video.map(fo).where(f, np.nan)
    gap = np.zeros(len(x), dtype=bool)
    for v, g in x.groupby("video"):
        t = np.sort(g.loc[g.role_seg == "seen", "frame"].to_numpy())
        fi = g.index[g.role_seg == "fit"]
        if len(t) and len(fi):
            fr = x.loc[fi, "frame"].to_numpy()
            i = np.searchsorted(t, fr)
            dmin = np.minimum(np.abs(fr - t[np.clip(i - 1, 0, len(t) - 1)]), np.abs(t[np.clip(i, 0, len(t) - 1)] - fr))
            gap[x.index.get_indexer(fi)] = dmin <= dc.GAP
    x["gap25"] = gap
    ood = u[u.label == ood_class][["file", "video", "frame", "label"]]
    return x, ood, len(vs_ood), len(unseen), len(ckpt)


def kvasir():
    cache, stage = OUT / "kvasir_images.csv.gz", REPO / "data/staged/p2d_kvasir_256"
    if cache.exists() and len(list(stage.glob("*.tar"))) == 14 and not list(stage.glob("*.part")):
        A = pd.read_csv(cache, dtype={"file": str, "folder": str, "video_a": str, "video_b": str})
    else:
        A = kvasir_read()
        A.to_csv(cache, index=False)
    S = dc.load()
    S["file"] = S.filename.str[:-4]
    multi = set(S.filename[S.filename.duplicated(keep=False)].str[:-4])
    a_ok = bool(A.video_b.notna().all() and (A.video_a == A.video_b).all() and (A.frame_a == A.frame_b).all())
    J = S.drop_duplicates("file").merge(A, on="file", how="left", suffixes=("", "_arc"))
    three = {"archive_images": len(A), "archive_rsplit_equals_regex": a_ok,
             "split_rows_found_in_archive": int(J.folder.notna().sum()), "split_unique_files": int(S.file.nunique()),
             "video_equal_split_vs_archive": bool((J.dropna(subset=["folder"]).video_a_arc == J.dropna(subset=["folder"]).video_a).all())
             if "video_a_arc" in J else None}
    J2 = S.merge(A[["file", "video_a"]].rename(columns={"video_a": "video_c"}), on="file", how="inner")
    three["official_folds_with_archive_ids"] = {k: v for k, v in dc.both_folds(J2, "video_c").items() if k != "ids"}
    A["label"] = A.folder.map(FOLDER2LABEL)
    lab_ok = J.dropna(subset=["folder"])
    three["split_label_equals_archive_folder"] = float((lab_ok.label == lab_ok.folder.map(FOLDER2LABEL)).mean())
    dup = A.groupby("pix").agg(n=("file", "size"), videos=("video_a", "nunique"))
    U = A[A.folder.isin(FOLDER2LABEL) & ~A.file.isin(multi)].rename(columns={"video_a": "video", "frame_a": "frame"})
    info = {"three_way_parse": three, "dropped_classes": {c: int((A.folder == c).sum()) for c in DROPPED},
            "two_label_files_dropped": len(multi), "id_pool_frames": len(U), "videos": int(U.video.nunique()),
            "image_modes": A["mode"].value_counts().to_dict(), "image_formats": A["format"].value_counts().to_dict(), "sizes": (A.w.astype(str) + "x" + A.h.astype(str)).value_counts().to_dict(),
            "duplicate_pixels": {"hashes_with_gt1_files": int((dup.n > 1).sum()), "files_in_them": int(dup.n[dup.n > 1].sum()),
                                 "hashes_across_videos": int((dup.videos > 1).sum())}, "candidates": {}}
    sets = []
    for c in ("Angiectasia", "Erosion"):
        x, ood, nv_ood, nv_un, nv_ck = kvasir_build(U, c)
        ref = json.loads((OUT / "design_check.json").read_text())["gap"]
        assert int((x.role_lit == "fit").sum()) == ref["frame_level_split"][c]["fit_frames"], c
        assert int((x.role_seg == "fit").sum()) == ref["segment_level_split"][c]["fit_frames"], c
        fit_seg = x[x.role_seg == "fit"]
        seen_seg = x[x.role_seg == "seen"]
        vids_gap = set(fit_seg.loc[~fit_seg.gap25, "video"])
        info["candidates"][c] = {
            "ood_videos": nv_ood, "ood_frames": len(ood), "unseen_videos": nv_un, "unseen_frames": int((x.role_lit == "unseen").sum()),
            "ckpt_videos": nv_ck, "id_classes": sorted(x.label.unique()),
            "K_lit": {r: int((x.role_lit == r).sum()) for r in ("fit", "val", "seen")} | {
                "fit_videos": int(x.loc[x.role_lit == "fit", "video"].nunique()),
                "afit_fold_videos": x[x.role_lit == "fit"].groupby("afit_fold_lit").video.nunique().astype(int).to_dict()},
            "K_seg": {r: int((x.role_seg == r).sum()) for r in ("fit", "val", "seen")} | {
                "fit_videos": int(fit_seg.video.nunique()),
                "afit_fold_videos": fit_seg.groupby("afit_fold_seg").video.nunique().astype(int).to_dict(),
                "gap25_removed": int(fit_seg.gap25.sum()), "gap25_removed_share": round(float(fit_seg.gap25.mean()), 4),
                "seen_with_fit_video_nogap": int(seen_seg.video.isin(set(fit_seg.video)).sum()),
                "seen_with_fit_video_gap": int(seen_seg.video.isin(vids_gap).sum())},
            "G_leak_K_lit": round(float(x.loc[x.role_lit == "seen", "video"].isin(set(x.loc[x.role_lit == "fit", "video"])).mean()), 4)}
        sets.append(pd.concat([x.assign(ood_candidate=c, role_ood=""), ood.assign(ood_candidate=c, role_lit="ood", role_seg="ood")]))
    pd.concat(sets)[["ood_candidate", "file", "video", "frame", "label", "role_lit", "role_seg", "seg", "afit_fold_lit",
                     "afit_fold_seg", "gap25"]].to_csv(OUT / "kvasir_sets.csv.gz", index=False)
    (OUT / "kvasir_phase0.json").write_text(json.dumps(info, indent=1, default=int) + "\n")
    print(json.dumps(info, indent=1, default=int))


def brain():
    import h5py
    stage = REPO / "data/staged/p2d_brain_256"
    stage.mkdir(parents=True, exist_ok=True)
    rows, fields = [], set()
    from PIL import Image
    with tarfile.open(stage / "brain.tar.part", "w") as to:
        for zp in sorted((REPO / "data/raw/brain_tumor_cheng").glob("brainTumorDataPublic_*.zip")):
            z = zipfile.ZipFile(zp)
            for n in z.namelist():
                if not n.endswith(".mat"):
                    continue
                c = h5py.File(io.BytesIO(z.read(n)), "r")["cjdata"]
                fields |= set(c.keys())
                im = np.array(c["image"]).T.astype(np.float64)
                lo, hi = im.min(), im.max()
                u8 = np.uint8(255.0 / (hi - lo) * (im - lo)) if hi > lo else np.zeros_like(im, dtype=np.uint8)
                pid = "".join(chr(int(v)) for v in np.array(c["PID"]).ravel())
                rows.append({"idx": int(n[:-4]), "pid": pid, "label": int(np.array(c["label"]).ravel()[0]),
                             "h": im.shape[0], "w": im.shape[1], "pix": hashlib.sha1(np.array(c["image"]).tobytes()).hexdigest()})
                add(to, f"{int(n[:-4])}.png", stage_png(Image.fromarray(u8), 256, "L"))
    (stage / "brain.tar.part").rename(stage / "brain.tar")
    B = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    B["cls"] = B.label.map(BRAIN)
    spp = B.groupby("pid").size()
    ppc = B.groupby("pid").cls.nunique()
    dup = B.groupby("pix").agg(n=("idx", "size"), pids=("pid", "nunique"))
    info = {"fields": sorted(fields), "slice_index_field": False, "n_slices": len(B), "n_patients": int(B.pid.nunique()),
            "slices_per_class": B.cls.value_counts().to_dict(), "patients_per_class": B.groupby("cls").pid.nunique().to_dict(),
            "patients_with_gt1_class": int((ppc > 1).sum()), "pid_parse_empty": int((B.pid.str.len() == 0).sum()),
            "slices_per_patient": {k: int(np.quantile(spp.to_numpy(), p)) for k, p in (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1))} | {"mean": round(float(spp.mean()), 2)},
            "sizes": (B.w.astype(str) + "x" + B.h.astype(str)).value_counts().to_dict(),
            "duplicate_pixels": {"hashes_with_gt1_slices": int((dup.n > 1).sum()), "slices_in_them": int(dup.n[dup.n > 1].sum()),
                                 "hashes_across_patients": int((dup.pids > 1).sum())},
            "G_audit_sanity": bool(len(B) == 3064 and B.pid.nunique() == 233 and B.cls.value_counts().to_dict() ==
                                   {"glioma": 1426, "pituitary": 930, "meningioma": 708}), "candidates": {}}
    sets = []
    for c in ("meningioma", "pituitary", "glioma"):
        x = B[B.cls != c].copy()
        pc = x.groupby("pid").cls.agg(lambda s: s.value_counts().index[0])
        rng = np.random.default_rng(0)
        unseen = set()
        for _, g in pc.groupby(pc):
            g = np.array(sorted(g.index))
            unseen |= set(g[rng.permutation(len(g))[: int(round(0.2 * len(g)))]])
        rest = np.array(sorted(p for p in x.pid.unique() if p not in unseen))
        ckpt = set(rest[rng.permutation(len(rest))[: int(round(0.1 * len(rest)))]])
        x["role"] = np.where(x.pid.isin(unseen), "unseen", np.where(x.pid.isin(ckpt), "ckpt", ""))
        y = x[x.role == ""]
        role = np.empty(len(y), dtype=object)
        for _, idx in y.groupby("cls").indices.items():
            p = rng.permutation(idx)
            n_tr, n_va = int(round(0.60 * len(p))), int(round(0.20 * len(p)))
            role[p[:n_tr]], role[p[n_tr:n_tr + n_va]], role[p[n_tr + n_va:]] = "fit", "val", "seen"
        x.loc[y.index, "role"] = role
        f = x.role == "fit"
        fo = assign_folds(x.loc[f, "pid"].to_numpy(), x.loc[f, "cls"].to_numpy(), np.random.default_rng(0))
        x["afit_fold"] = x.pid.map(fo).where(f, np.nan)
        o = B[B.cls == c].assign(role="ood")
        info["candidates"][c] = {"id_classes": sorted(x.cls.unique()), "ood_slices": len(o), "ood_patients": int(o.pid.nunique()),
                                 **{r: int((x.role == r).sum()) for r in ("fit", "val", "seen", "unseen", "ckpt")},
                                 "unseen_patients": len(unseen), "ckpt_patients": len(ckpt),
                                 "fit_patients": int(x.loc[f, "pid"].nunique()),
                                 "afit_fold_patients": x[f].groupby("afit_fold").pid.nunique().astype(int).to_dict(),
                                 "G_leak": round(float(x.loc[x.role == "seen", "pid"].isin(set(x.loc[f, "pid"])).mean()), 4)}
        sets.append(pd.concat([x, o]).assign(ood_candidate=c))
    B.to_csv(OUT / "brain_images.csv.gz", index=False)
    pd.concat(sets)[["ood_candidate", "idx", "pid", "cls", "role", "afit_fold"]].to_csv(OUT / "brain_sets.csv.gz", index=False)
    (OUT / "brain_phase0.json").write_text(json.dumps(info, indent=1, default=int) + "\n")
    print(json.dumps(info, indent=1, default=int))


if __name__ == "__main__":
    {"kvasir": kvasir, "brain": brain}[sys.argv[1]]()
