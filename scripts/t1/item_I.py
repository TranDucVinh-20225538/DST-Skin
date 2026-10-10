#!/usr/bin/env python3
"""T1 item I: inventory, cell registry and the Camelyon17 manifest check (MANIFEST_OK). CPU only.

Outputs results/t1/I/: registry.csv, inventory.md, inventory_files.csv, camelyon_manifest.csv,
camelyon_manifest.md, MANIFEST_OK, manifest_checks.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as CM  # noqa: E402

OUT = CM.RES / "I"
REPO = CM.REPO
FM_NAMES = set(CM.FMS)
DOC = {"train": 302436, "id_val": 33560, "ood": 85054}


def npy_shape(npz_path, key):
    with zipfile.ZipFile(npz_path) as z:
        with z.open(key + ".npy") as f:
            v = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format._read_array_header(f, v)
    return shape


def npz_keys(npz_path):
    with zipfile.ZipFile(npz_path) as z:
        return [n[:-4] for n in z.namelist()]


def parse_cell(cell):
    ds = cell.split("_")[0]
    rest = cell[len(ds) + 1:]
    m = re.match(r"(.+?)_s(\d+)(_.*)?$", rest)
    bb, seed, tail = m.group(1), int(m.group(2)), (m.group(3) or "")
    core_bb = bb[3:] if bb.startswith("fm_") else bb
    fam = "FM" if (bb.startswith("fm_") or core_bb in FM_NAMES) else "CNN"
    return ds, bb, fam, seed, tail.lstrip("_")


def fold_stats(cell, seed):
    from crossfit_ood import core as C
    z = np.load(CM.INPUTS / f"{cell}.npz", allow_pickle=False)
    gtr = z["groups_train"]
    gid = z["groups_id_eval"]
    if "fold_train" in z.files:
        fm = C._fold_ids_to_maps(z["fold_train"].astype(int), gtr)[0]
        src = "fold_train"
    else:
        gid_used = gid[np.isin(gid, gtr)]
        fm = C._new_fold_maps("paper_2fold", gtr, gid_used, 2, 1, seed)[0]
        src = "package _new_fold_maps(seed)"
    tf = C._apply_map(fm, gtr)
    res = {"fold_source": src}
    for f in (0, 1):
        g = gtr[tf == f]
        u, c = np.unique(g, return_counts=True)
        res[f"n_groups_fit_f{f}"] = len(u)
        res[f"N_fit_f{f}"] = int(len(g))
        res[f"gsize_f{f}"] = f"{c.min()}/{int(np.median(c))}/{c.max()}"
    res["fold_map"] = {str(k): int(v) for k, v in fm.items()}
    return res


def registry():
    pci = list(csv.DictReader(open(REPO / "results/r3/1/paper_ci_v2.csv")))
    delta = {}
    for r in pci:
        delta.setdefault(r["cell"], {})[r["scorer"]] = r
    meta = {}
    for f in ("cells_camelyon.csv", "cells_med.csv", "cells_medfm.csv"):
        for r in csv.DictReader(open(CM.W1 / f)):
            meta[r["cell"]] = r
    rows, folds = [], {}
    cells = sorted(Path(p).stem for p in glob.glob(str(CM.INPUTS / "*.npz")) if not p.endswith("_head.npz"))
    for cell in cells:
        p = CM.INPUTS / f"{cell}.npz"
        keys = npz_keys(p)
        ds, bb, fam, seed, tail = parse_cell(cell)
        fs = fold_stats(cell, seed)
        folds[cell] = fs
        pc = [str(x) for x in (CM.W1 / "out_v2" / f"{cell}_maha" / "paper_ci.csv",
                                CM.W1 / "out_v2" / f"{cell}_knnmc" / "paper_ci.csv",
                                CM.W1 / "out_v2" / cell / "paper_ci.csv") if x.exists()]
        dm = delta.get(cell, {})
        rows.append(dict(
            cell_id=cell, dataset=ds, backbone=bb, family=fam, seed=seed,
            d=npy_shape(p, "features_train")[1], input_npz=str(p), paper_ci_csv=";".join(pc) if pc else "NA",
            id_acc=meta.get(cell, {}).get("id_acc", "NA"),
            n_groups_fit_f0=fs["n_groups_fit_f0"], n_groups_fit_f1=fs["n_groups_fit_f1"],
            N_fit_f0=fs["N_fit_f0"], N_fit_f1=fs["N_fit_f1"],
            has_labels="labels_train" in keys, has_head=(CM.INPUTS / f"{cell}_head.npz").exists(),
            has_delta_mahalanobis_l2="mahalanobis_l2" in dm, has_delta_knn_mean_cosine="knn_mean_cosine" in dm,
            notes=";".join(x for x in (f"variant={tail}" if tail else "", f"folds={fs['fold_source']}",
                                        f"gsize_f0={fs['gsize_f0']}", f"gsize_f1={fs['gsize_f1']}",
                                        f"seen={dm['mahalanobis_l2']['seen']}" if "mahalanobis_l2" in dm else "") if x)))
    with open(OUT / "registry.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT / "registry.csv.tmp", OUT / "registry.csv")
    return rows, folds


def _file_row(p):
    st = os.stat(p)
    return dict(path=str(p), size=st.st_size, mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
                sha256=CM.sha256_file(p))


def inventory_files(n_workers):
    pats = {
        "r3_item1_inputs": [str(CM.INPUTS / "*.npz")],
        "r3_item1_paper_ci": [str(CM.W1 / "out_v2" / "*" / "paper_ci.csv"), str(REPO / "results/r3/1/paper_ci_v2.csv")],
        "foundation_gate_feats": [str(REPO / "outputs/rigor_pack/foundation_gate/feats/camelyon_*.npz")],
        "cnn_feature_caches": [str(REPO / "outputs/features/camelyon17/**/*")],
        "miccai_scores": [str(REPO / "outputs/rigor_pack/miccai_campaign/scores/camelyon_*.npz")],
        "slide_identity": [str(REPO / "outputs/reports/rigor_pack/miccai_campaign/trackA_foundation/slide_identity.csv")],
        "r3_item3_whitening": [str(REPO / "outputs/rigor_pack/r3/item3/*.json")],
        "r3_item6_dose": [str(REPO / "results/r3/6/dose.csv")],
        "paper3_caches": [str(CM.HOME / "Downloads/**/*_z.npz")],
        "retrain_arm": [str(REPO / "scripts/r3/item5_b_isbi_patch2.py"), str(REPO / "decisions/precommit_isbi_patch2_2026-10-06.md"),
                        str(REPO / "logs/isbi_patch*"), str(REPO / "logs/*p2d*"), str(CM.HOME / "r3work/p2d_*.log")],
        "camelyon_metadata": [str(REPO / "data/raw/wilds/camelyon17_v1.0/metadata.csv")],
    }
    todo = []
    for cat, ps in pats.items():
        found = sorted({f for p in ps for f in glob.glob(p, recursive=True) if os.path.isfile(f)})
        todo += [(cat, f) for f in found]
        if not found:
            todo.append((cat, None))
    with ProcessPoolExecutor(n_workers) as ex:
        futs = [(cat, f, ex.submit(_file_row, f) if f else None) for cat, f in todo]
        rows = []
        for cat, f, fu in futs:
            rows.append(dict(category=cat, **(fu.result() if fu else dict(path="NA", size="NA", mtime="NA", sha256="NA"))))
    with open(OUT / "inventory_files.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "path", "size", "mtime", "sha256"])
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT / "inventory_files.csv.tmp", OUT / "inventory_files.csv")
    return rows


def manifest(reg_rows, folds):
    import importlib.util
    sys.path.insert(0, str(REPO / "scripts/rigor"))
    spec = importlib.util.spec_from_file_location("rigor_common", REPO / "scripts/rigor/common.py")
    RC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RC)
    meta = RC.load_camelyon_metadata(REPO)
    checks, warns = {}, []
    sp = meta["wilds_split"].to_numpy()
    slide = meta["slide"].astype(str).to_numpy()
    center = meta["center"].to_numpy()
    patient = meta["patient"].astype(str).to_numpy()
    tr, iv, te = sp == 0, sp == 1, sp == 2
    train_slides = sorted(set(slide[tr]), key=int)
    hosp = {s: sorted(set(center[slide == s].tolist())) for s in train_slides}
    rows = [dict(slide=s, hospital=hosp[s][0] if len(hosp[s]) == 1 else "|".join(map(str, hosp[s])),
                 n_train_patches=int((tr & (slide == s)).sum()), n_idval_patches=int((iv & (slide == s)).sum()))
            for s in train_slides]
    with open(OUT / "camelyon_manifest.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slide", "hospital", "n_train_patches", "n_idval_patches"])
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT / "camelyon_manifest.csv.tmp", OUT / "camelyon_manifest.csv")
    per_h = {}
    for r in rows:
        per_h.setdefault(str(r["hospital"]), []).append(r)
    checks["one_hospital_per_slide"] = all(len(h) == 1 for h in hosp.values())
    checks["n_train_slides"] = len(train_slides)
    checks["train_hospitals"] = sorted(per_h)
    checks["slides_per_hospital"] = {h: len(v) for h, v in per_h.items()}
    c1 = (checks["one_hospital_per_slide"] and len(train_slides) == 30 and sorted(per_h) == ["0", "3", "4"]
          and all(len(v) == 10 for v in per_h.values()))
    checks["C1_30_slides_10_per_hospital_034"] = c1
    ood_centers = sorted(set(center[te].tolist()))
    ood_slides, ood_patients = set(slide[te]), sorted(set(patient[te]))
    train_patients = set(patient[tr])
    checks["ood_centers"] = ood_centers
    checks["ood_n_patients"] = len(ood_patients)
    checks["ood_slides_in_train"] = sorted(ood_slides & set(train_slides))
    checks["ood_patients_in_train"] = sorted(set(ood_patients) & train_patients)
    c2 = ood_centers == [2] and not checks["ood_slides_in_train"] and not checks["ood_patients_in_train"]
    checks["C2_ood_hospital2_disjoint"] = c2
    idval_centers = sorted(set(center[iv].tolist()))
    checks["id_val_centers"] = idval_centers
    checks["id_val_slides_subset_of_train"] = set(slide[iv]) <= set(train_slides)
    totals = dict(train=int(tr.sum()), id_val=int(iv.sum()), ood=int(te.sum()))
    checks["totals"] = totals
    for k, v in DOC.items():
        if totals[k] != v:
            warns.append(f"WARN total {k} = {totals[k]} vs documented {v}")
    ntr = {r["slide"]: r["n_train_patches"] for r in rows}
    checks["per_slide_train_min_max"] = [min(ntr.values()), max(ntr.values())]
    # caches
    cam = [r for r in reg_rows if r["dataset"] == "camelyon"]
    cache_ok, fold_ok, cache_detail, fold_detail = True, True, {}, {}
    hosp_of = {r["slide"]: str(r["hospital"]) for r in rows}
    for r in cam:
        z = np.load(r["input_npz"], allow_pickle=False)
        gtr, gid = z["groups_train"].astype(str), z["groups_id_eval"].astype(str)
        n_ood = npy_shape(r["input_npz"], "features_ood")[0]
        s_tr = set(gtr)
        u, c = np.unique(gtr, return_counts=True)
        cnt_match = all(ntr.get(s) == int(n) for s, n in zip(u, c))
        ok = (s_tr == set(train_slides)) and set(gid) <= set(train_slides)
        cache_detail[r["cell_id"]] = dict(slide_set_equal=s_tr == set(train_slides), idval_subset=set(gid) <= set(train_slides),
                                          per_slide_counts_equal_metadata=cnt_match, n_train=len(gtr), n_id_eval=len(gid), n_ood=n_ood)
        if not cnt_match:
            warns.append(f"WARN {r['cell_id']}: per-slide train counts differ from metadata")
        if len(gtr) != DOC["train"] or len(gid) != DOC["id_val"] or n_ood != DOC["ood"]:
            warns.append(f"WARN {r['cell_id']}: cache sizes {len(gtr)}/{len(gid)}/{n_ood} vs documented")
        cache_ok &= ok
        fm = folds[r["cell_id"]]["fold_map"]
        per = {}
        for f in (0, 1):
            per[f] = {h: sum(1 for s, ff in fm.items() if ff == f and hosp_of.get(s) == h) for h in ("0", "3", "4")}
            per[f]["other"] = sum(1 for s, ff in fm.items() if ff == f and hosp_of.get(s) not in ("0", "3", "4"))
        fold_detail[r["cell_id"]] = per
        fold_ok &= all(per[f][h] == 5 for f in (0, 1) for h in ("0", "3", "4")) and all(per[f]["other"] == 0 for f in (0, 1))
    checks["C3_cache_slide_sets_equal_metadata"] = cache_ok
    checks["C4_every_fold_5_per_hospital"] = fold_ok
    checks["n_camelyon_cells"] = len(cam)
    manifest_ok = bool(c1 and c2 and cache_ok and fold_ok and len(cam) > 0)
    checks["MANIFEST_OK"] = manifest_ok
    CM.atomic_write_text(OUT / "MANIFEST_OK", "true\n" if manifest_ok else "false\n")
    CM.atomic_write_text(OUT / "manifest_checks.json", json.dumps(dict(checks=checks, warnings=warns,
                                                                      cache_detail=cache_detail, fold_detail=fold_detail), indent=1))
    L = ["# Camelyon17 manifest (item I)", "",
         f"Source: `data/raw/wilds/camelyon17_v1.0/metadata.csv` via `scripts/rigor/common.py:load_camelyon_metadata` (R3 helper used by `scripts/r3/item6_dose.py`).", "",
         "| hospital | train slides | train patches | id_val patches |", "|---|---|---|---|"]
    for h in sorted(per_h):
        L.append(f"| {h} | {len(per_h[h])} | {sum(r['n_train_patches'] for r in per_h[h])} | {sum(r['n_idval_patches'] for r in per_h[h])} |")
    L += ["", f"OOD (`test`) split: hospitals {ood_centers}, {totals['ood']} patches, {len(ood_patients)} patients; "
          f"OOD slides among training slides: {checks['ood_slides_in_train'] or 'none'}; OOD patients among training patients: {checks['ood_patients_in_train'] or 'none'}.",
          f"id_val hospitals: {idval_centers}; id_val slides subset of the 30 training slides: {checks['id_val_slides_subset_of_train']}.",
          f"Totals train / id_val / OOD: {totals['train']} / {totals['id_val']} / {totals['ood']} (documented 302,436 / 33,560 / 85,054).",
          f"Per-slide train patches min / max: {checks['per_slide_train_min_max'][0]} / {checks['per_slide_train_min_max'][1]}.", "",
          "## paper_2fold slides per hospital per fold (every Camelyon cell)", "",
          "| cell | fold 0 (h0/h3/h4/other) | fold 1 (h0/h3/h4/other) |", "|---|---|---|"]
    for c, per in fold_detail.items():
        L.append(f"| {c} | {per[0]['0']}/{per[0]['3']}/{per[0]['4']}/{per[0]['other']} | {per[1]['0']}/{per[1]['3']}/{per[1]['4']}/{per[1]['other']} |")
    L += ["", "## Checks", "",
          f"- 30 training slides, exactly 10 per hospital over {{0, 3, 4}}: **{c1}**",
          f"- OOD all hospital 2, no OOD slide or patient among the training slides: **{c2}**",
          f"- every Camelyon cache's slide set equals the metadata's: **{cache_ok}** ({len(cam)} cells)",
          f"- every fold of every Camelyon cell holds exactly 5 slides per hospital: **{fold_ok}**",
          f"- **MANIFEST_OK = {str(manifest_ok).lower()}**", ""]
    L += ["## Warnings (not stops)", ""] + ([f"- {w}" for w in warns] or ["- none"])
    CM.atomic_write_text(OUT / "camelyon_manifest.md", "\n".join(L) + "\n")
    return manifest_ok, checks, warns


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    reg, folds = registry()
    print("registry", len(reg), time.time() - t0, flush=True)
    mok, checks, warns = manifest(reg, folds)
    print("MANIFEST_OK", mok, flush=True)
    inv = inventory_files(nw)
    print("inventory files", len(inv), time.time() - t0, flush=True)
    CM.atomic_write_text(OUT / "wall.json", json.dumps({"wall_s": time.time() - t0}))


if __name__ == "__main__":
    main()
