#!/usr/bin/env python3
"""R3 item 8 / P2-a Phase 0 (results/r3/8/PRECOMMIT.json): access, audit sanity, ChestMNIST audit, leak gate.

1. Stage: every NIH ChestX-ray14 image (12 archives) -> grayscale, 256 x 256 PNG in data/staged/nih_256/{archive}.tar
   (kept for P2-a), and a 224 x 224 uint8 copy in $TMPDIR (audit only, deleted on exit).
2. Audit sanity: patient-ID parse rate, patient overlap of the official train_val / test lists (must be 0).
3. ChestMNIST (MedMNIST+ 224) -> NIH image IDs: nearest neighbour on 32 x 32 block means (Pearson = cosine of
   centred, normalised vectors), accepted if the Pearson correlation at 224 x 224 is >= 0.98. Label agreement with
   Data_Entry is reported as a check of the mapping.
4. Leak rates: ChestMNIST test (and val) images whose patient has >= 1 ChestMNIST train image; the Wang et al. 2017
   image-level random 70 / 10 / 20 split (seed 0, stratified on 'No Finding'), same definition on its test set.
   M_std = ChestMNIST split if its train->test leak rate >= 5%, else the Wang split. G_leak on M_std (< 5% -> STOP).
Writes results/r3/8/p2a/phase0.json, phase0_chestmnist_map.csv.gz, wang_split.csv.gz
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/nih_cxr14"
STAGE = REPO / "data/staged/nih_256"
OUT = REPO / "results/r3/8/p2a"
TMP = Path(os.environ.get("TMPDIR", "/tmp"))
CORR_MIN, LEAK_MIN = 0.98, 0.05
CM_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
             "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def stage_archive(arc: str):
    from PIL import Image
    names, a224 = [], []
    out = STAGE / arc.replace(".tar.gz", ".tar")
    tmp = out.with_suffix(".tar.part")
    with tarfile.open(RAW / arc) as tf, tarfile.open(tmp, "w") as to:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".png")):
                continue
            im = Image.open(tf.extractfile(m)).convert("L")
            name = Path(m.name).name
            b = io.BytesIO()
            im.resize((256, 256), Image.BICUBIC).save(b, format="PNG")
            ti = tarfile.TarInfo(name)
            ti.size = b.tell()
            b.seek(0)
            to.addfile(ti, b)
            names.append(name)
            a224.append(np.asarray(im.resize((224, 224), Image.BICUBIC), dtype=np.uint8))
    tmp.rename(out)
    np.save(TMP / (arc + ".224.npy"), np.stack(a224))
    return arc, names


def block32(x: np.ndarray) -> np.ndarray:
    v = x.reshape(len(x), 32, 7, 32, 7).mean(axis=(2, 4), dtype=np.float32).reshape(len(x), -1)
    v -= v.mean(1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float32)
    b = b.reshape(len(b), -1).astype(np.float32)
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def leak_rate(pid_train, pid_eval) -> float:
    s = set(pid_train)
    return float(np.mean([p in s for p in pid_eval])) if len(pid_eval) else float("nan")


def main() -> int:
    import torch
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    res = {"stage": {}}
    arcs = sorted(p.name for p in RAW.glob("images_*.tar.gz"))
    res["G0_access"] = {"archives": len(arcs), "all_verified": all((RAW / (a + ".ok")).exists() for a in arcs)}
    with ProcessPoolExecutor(max_workers=min(12, int(os.environ.get("SLURM_CPUS_PER_TASK", "12")))) as ex:
        staged = dict(ex.map(stage_archive, arcs))
    names = [n for a in arcs for n in staged[a]]
    nih224 = np.concatenate([np.load(TMP / (a + ".224.npy")) for a in arcs])
    res["stage"] = {"n_images": len(names), "staged_dir": str(STAGE.relative_to(REPO)), "seconds": round(time.time() - t0)}
    print("staged", len(names), time.time() - t0, flush=True)

    d = pd.read_csv(RAW / "Data_Entry_2017_v2020.csv").set_index("Image Index")
    pid_fn = pd.Series({n: int(n.split("_")[0]) for n in names})
    tv = set(open(RAW / "train_val_list.txt").read().split())
    te = set(open(RAW / "test_list.txt").read().split())
    p_tv, p_te = set(d.loc[d.index.isin(tv), "Patient ID"]), set(d.loc[d.index.isin(te), "Patient ID"])
    res["G_audit_sanity"] = {
        "n_metadata_rows": len(d), "n_images_in_archives": len(names),
        "images_in_both": int(len(set(names) & set(d.index))),
        "patient_id_parse_rate": float((pid_fn == d.loc[pid_fn.index, "Patient ID"]).mean()),
        "official_train_val": len(tv), "official_test": len(te),
        "official_patient_overlap": len(p_tv & p_te), "n_patients": int(d["Patient ID"].nunique())}
    res["G_audit_sanity"]["pass"] = bool(res["G_audit_sanity"]["official_patient_overlap"] == 0 and
                                         res["G_audit_sanity"]["patient_id_parse_rate"] >= 0.99)

    z = np.load(RAW / "chestmnist_224.npz")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    R = torch.from_numpy(block32(nih224))
    rows = []
    for split in ("train", "val", "test"):
        X, Y = z["%s_images" % split], z["%s_labels" % split]
        Q = torch.from_numpy(block32(X))
        best = np.empty(len(X), dtype=np.int64)
        for i in range(0, len(X), 4096):
            best[i:i + 4096] = (Q[i:i + 4096] @ R.T).argmax(1).numpy()
        corr = np.concatenate([pearson(X[i:i + 4096], nih224[best[i:i + 4096]]) for i in range(0, len(X), 4096)])
        for j in range(len(X)):
            n = names[best[j]]
            rows.append({"split": split, "idx": j, "nih_image": n, "corr224": float(corr[j]),
                         "patient": int(d.loc[n, "Patient ID"]),
                         "labels_cm": "".join(map(str, Y[j])),
                         "labels_nih": "".join("1" if l in d.loc[n, "Finding Labels"].split("|") else "0" for l in CM_LABELS)})
        print("matched", split, len(X), time.time() - t0, flush=True)
    M = pd.DataFrame(rows)
    M["accepted"] = M.corr224 >= CORR_MIN
    M.to_csv(OUT / "phase0_chestmnist_map.csv.gz", index=False)
    A = M[M.accepted]
    dup = int(A.nih_image.duplicated().sum())
    cm = {"n": len(M), "n_accepted": int(len(A)), "accept_rate": float(M.accepted.mean()),
          "corr224_quantiles_all": [float(x) for x in M.corr224.quantile([0.001, 0.01, 0.5])],
          "duplicate_nih_targets_among_accepted": dup,
          "label_agreement_accepted": float((A.labels_cm == A.labels_nih).mean()),
          "n_per_split": M.split.value_counts().to_dict()}
    tr = A[A.split == "train"].patient
    cm["leak_rate_train_to_test"] = leak_rate(tr, A[A.split == "test"].patient)
    cm["leak_rate_train_to_val"] = leak_rate(tr, A[A.split == "val"].patient)
    cm["test_vs_official_test_overlap"] = float(A[A.split == "test"].nih_image.isin(te).mean())
    res["chestmnist_audit"] = cm

    from sklearn.model_selection import train_test_split
    dd = d.reset_index()
    nf = (dd["Finding Labels"] == "No Finding").astype(int)
    tr_i, rest = train_test_split(np.arange(len(dd)), train_size=0.7, random_state=0, stratify=nf)
    va_i, te_i = train_test_split(rest, train_size=1 / 3, random_state=0, stratify=nf.iloc[rest])
    role = np.empty(len(dd), dtype=object)
    role[tr_i], role[va_i], role[te_i] = "train", "val", "test"
    W = pd.DataFrame({"image": dd["Image Index"], "patient": dd["Patient ID"], "role": role})
    W.to_csv(OUT / "wang_split.csv.gz", index=False)
    wtr = W[W.role == "train"].patient
    res["wang_split"] = {"sizes": W.role.value_counts().to_dict(),
                         "leak_rate_train_to_test": leak_rate(wtr, W[W.role == "test"].patient),
                         "leak_rate_train_to_val": leak_rate(wtr, W[W.role == "val"].patient)}

    if cm["accept_rate"] >= 0.99 and cm["leak_rate_train_to_test"] >= LEAK_MIN:
        m_std, leak = "chestmnist", cm["leak_rate_train_to_test"]
    elif cm["accept_rate"] >= 0.99:
        m_std, leak = "wang2017", res["wang_split"]["leak_rate_train_to_test"]
    else:
        m_std, leak = "undetermined (ChestMNIST mapping accept rate < 0.99; pause, report)", float("nan")
    res["M_std"] = m_std
    res["G_leak"] = {"leak_rate": leak, "threshold": LEAK_MIN,
                     "pass": bool(leak >= LEAK_MIN) if leak == leak else None,
                     "decision": ("continue to training" if leak >= LEAK_MIN else
                                  "STOP P2-a after the leak audit (clean - not tested)") if leak == leak else "pause"}
    res["seconds"] = round(time.time() - t0)
    (OUT / "phase0.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
