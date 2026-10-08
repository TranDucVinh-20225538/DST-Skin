#!/usr/bin/env python3
"""R3 item 8 / P2-a set definitions and OOD staging (results/r3/8/PRECOMMIT.json; M_std = wang2017 by Phase 0).

Sets, from results/r3/8/p2a/wang_split.csv.gz (image-level 70 / 10 / 20, seed 0):
  ckpt   = all Wang-train images of 10% of the Wang-train patients (default_rng(0) permutation of the sorted patient
           IDs); checkpoint selection only (precommit: "10% patient-disjoint subset of the training patients")
  fit    = the remaining Wang-train images: CNN training set and the scorer fit set for every backbone
  seen   = Wang-test images whose patient has >= 1 fit image (ID_seen)
  unseen = Wang-test images whose patient has no Wang-train image (ID_unseen)
  test_ckpt = Wang-test images of ckpt patients (neither seen nor unseen; excluded, counted)
  wang_val  = unused
A-fit folds: medbench fold rule (medbench_phase0.assign_folds over fit patients, image class = 'No Finding',
default_rng(0)).
OOD staging: Shenzhen (all 662 PNGs) and Kermany pediatric CXR (ChestXRay2017.zip, all 5,856 images) are converted
with the NIH staging operation (grayscale, 256 x 256 bicubic) into data/staged/p2a_ood_256/{shenzhen,kermany_ped}.tar.
Writes results/r3/8/p2a/sets.csv.gz, sets.json; with --ood: ood.csv.gz, ood.json
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rigor"))
from medbench_phase0 import assign_folds  # noqa: E402

RAW = REPO / "data/raw"
OUT = REPO / "results/r3/8/p2a"
STAGE = REPO / "data/staged/p2a_ood_256"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def stage(name, items):
    from PIL import Image
    STAGE.mkdir(parents=True, exist_ok=True)
    out, tmp = STAGE / f"{name}.tar", STAGE / f"{name}.tar.part"
    rows = []
    with tarfile.open(tmp, "w") as to:
        for key, opener in items:
            with opener() as f:
                im = Image.open(f).convert("L")
                w, h = im.size
                b = io.BytesIO()
                im.resize((256, 256), Image.BICUBIC).save(b, format="PNG")
            ti = tarfile.TarInfo(key + ".png")
            ti.size = b.tell()
            b.seek(0)
            to.addfile(ti, b)
            rows.append({"source": name, "key": key, "orig_w": w, "orig_h": h})
    tmp.rename(out)
    return rows


def main() -> int:
    if sys.argv[1:] == ["--ood"]:
        return ood()
    W = pd.read_csv(OUT / "wang_split.csv.gz")
    d = pd.read_csv(RAW / "nih_cxr14/Data_Entry_2017_v2020.csv").set_index("Image Index")
    assert len(W) == len(d) == W.image.nunique()

    tr_p = np.array(sorted(W.loc[W.role == "train", "patient"].unique()))
    perm = np.random.default_rng(0).permutation(len(tr_p))
    ckpt_p = set(tr_p[perm[: int(round(0.1 * len(tr_p)))]].tolist())
    role = W.role.map({"train": "fit", "val": "wang_val", "test": "test"}).to_numpy(dtype=object)
    role[(W.role == "train").to_numpy() & W.patient.isin(ckpt_p).to_numpy()] = "ckpt"
    fit_p = set(W.loc[role == "fit", "patient"])
    te = role == "test"
    role[te & W.patient.isin(fit_p).to_numpy()] = "seen"
    role[te & W.patient.isin(ckpt_p).to_numpy()] = "test_ckpt"
    role[te & ~W.patient.isin(fit_p | ckpt_p).to_numpy()] = "unseen"
    S = pd.DataFrame({"image": W.image, "patient": W.patient, "role": role})
    S["no_finding"] = (d.loc[S.image, "Finding Labels"].to_numpy() == "No Finding").astype(int)
    fit = S[S.role == "fit"]
    fold_of = assign_folds(fit.patient.to_numpy(), fit.no_finding.to_numpy(), np.random.default_rng(0))
    S["afit_fold"] = [fold_of.get(p, -1) for p in S.patient]
    for f in FINDINGS:
        S[f] = [int(f in s.split("|")) for s in d.loc[S.image, "Finding Labels"]]
    S["view"] = d.loc[S.image, "View Position"].to_numpy()
    S.to_csv(OUT / "sets.csv.gz", index=False)

    vc = S.role.value_counts().to_dict()
    pat = {r: int(S.loc[S.role == r, "patient"].nunique()) for r in vc}
    info = {"m_std": "wang2017", "n_images": vc, "n_patients": pat,
            "n_wang_train_patients": int(len(tr_p)), "n_ckpt_patients": len(ckpt_p),
            "afit_fold_images": {int(k): int(v) for k, v in S.loc[S.role == "fit", "afit_fold"].value_counts().items()},
            "afit_fold_patients": {int(k): int(v) for k, v in pd.Series(fold_of).value_counts().items()},
            "seen_patients_in_fit": bool(set(S.loc[S.role == "seen", "patient"]) <= fit_p),
            "unseen_patients_disjoint": not (set(S.loc[S.role == "unseen", "patient"]) &
                                             set(W.loc[W.role == "train", "patient"])),
            "leak_rate_on_eval": vc.get("seen", 0) / (vc.get("seen", 0) + vc.get("unseen", 0))}
    (OUT / "sets.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


def ood() -> int:
    shz = sorted((RAW / "shenzhen/CXR_png").glob("CHNCXR_*.png"))
    z = zipfile.ZipFile(RAW / "medbench/kermany_v2/ChestXRay2017.zip")
    ker = sorted(n for n in z.namelist() if n.lower().endswith((".jpeg", ".jpg", ".png"))
                 and "__MACOSX" not in n and not Path(n).name.startswith("."))
    assert len(shz) == 662 and len(ker) == 5856, (len(shz), len(ker))
    O = pd.DataFrame(stage("shenzhen", [(p.stem, (lambda p=p: open(p, "rb"))) for p in shz]) +
                     stage("kermany_ped", [("kped_%05d" % i, (lambda n=n: z.open(n))) for i, n in enumerate(ker)]))
    O["member"] = ["" if s == "shenzhen" else ker[int(k.split("_")[1])] for s, k in zip(O.source, O.key)]
    O["shenzhen_tb"] = [int(k.endswith("_1")) if s == "shenzhen" else -1 for s, k in zip(O.source, O.key)]
    O.to_csv(OUT / "ood.csv.gz", index=False)
    info = {"shenzhen": int((O.source == "shenzhen").sum()), "shenzhen_tb": int((O.shenzhen_tb == 1).sum()),
            "shenzhen_normal": int((O.shenzhen_tb == 0).sum()), "kermany_ped": int((O.source == "kermany_ped").sum())}
    (OUT / "ood.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
