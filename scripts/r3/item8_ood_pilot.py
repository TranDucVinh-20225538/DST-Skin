#!/usr/bin/env python3
"""R3 item 8 / P2-a OOD ceiling pilot, as fixed in results/r3/8/pilot_plan.md (CPU, frozen DINOv2-B).

No same-group vs other-group fits: the pilot computes OOD AUROC only, never Delta.
Inputs (not in the repo): $PILOT_RAW/images_001.tar.gz (NIH ChestX-ray14 archive 1), $PILOT_RAW/shenzhen/*.png,
data/raw/medbench/kermany_v2/ChestXRay2017.zip. NIH images are extracted to $TMPDIR.
Output: results/r3/8/pilot_ood.csv, results/r3/8/pilot_ood.json
"""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RAW = Path(os.environ["PILOT_RAW"])
TMP = Path(os.environ.get("TMPDIR", "/tmp")) / "item8_pilot"
OUT = REPO / "results/r3/8"
N_EVAL, N_OOD, FIT_CAP = 300, 300, 1500


def nih_split():
    TMP.mkdir(parents=True, exist_ok=True)
    with tarfile.open(RAW / "images_001.tar.gz") as tf:
        mem = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".png")]
        tf.extractall(TMP, members=mem)
    paths = sorted(TMP.rglob("*.png"))
    pid = np.array([p.name.split("_")[0] for p in paths])
    pats = np.unique(pid)
    perm = np.random.default_rng(0).permutation(len(pats))
    fit_p = set(pats[perm[: len(pats) // 2]])
    is_fit = np.array([p in fit_p for p in pid])

    rng1 = np.random.default_rng(1)
    seen = []
    for p in sorted(fit_p):
        idx = np.flatnonzero(pid == p)
        if len(idx) >= 2:
            seen.append(int(rng1.choice(idx)))
    seen = sorted(rng1.choice(seen, min(N_EVAL, len(seen)), replace=False)) if len(seen) > N_EVAL else seen
    other = np.flatnonzero(~is_fit)
    unseen = sorted(np.random.default_rng(2).choice(other, min(N_EVAL, len(other)), replace=False))
    rest = np.setdiff1d(np.flatnonzero(is_fit), seen)
    fit = sorted(np.random.default_rng(3).choice(rest, min(FIT_CAP, len(rest)), replace=False))
    info = {"n_images": len(paths), "n_patients": int(len(pats)), "n_fit_patients": len(fit_p),
            "n_fit": len(fit), "n_seen": len(seen), "n_unseen": len(unseen)}
    return [paths[i] for i in fit], [paths[i] for i in seen], [paths[i] for i in unseen], info


def kermany_ped():
    z = zipfile.ZipFile(REPO / "data/raw/medbench/kermany_v2/ChestXRay2017.zip")
    names = sorted(n for n in z.namelist() if n.lower().endswith((".jpeg", ".jpg", ".png"))
                   and "__MACOSX" not in n and not Path(n).name.startswith("."))
    sel = sorted(np.random.default_rng(4).choice(len(names), N_OOD, replace=False))
    return [(z, names[i]) for i in sel], len(names)


def shenzhen():
    paths = sorted((RAW / "shenzhen").glob("*.png"))
    return paths, len(paths)


def load_img(x):
    from PIL import Image
    if isinstance(x, tuple):
        z, n = x
        return Image.open(io.BytesIO(z.read(n))).convert("RGB")
    return Image.open(x).convert("RGB")


def embed(items, model, tf, bs=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            x = torch.stack([tf(load_img(it)) for it in items[i:i + bs]])
            out.append(model.forward_features(x)["x_norm_clstoken"].float().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    import torch
    from torchvision import transforms as T
    from crossfit_ood import get_scorer

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    t0 = time.time()
    fit, seen, unseen, info = nih_split()
    ker, n_ker = kermany_ped()
    shz, n_shz = shenzhen()
    info.update({"kermany_ped_pool": n_ker, "kermany_ped_n": len(ker), "shenzhen_n": len(shz)})

    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=True).eval()
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    F = {k: embed(v, model, tf) for k, v in
         {"fit": fit, "seen": seen, "unseen": unseen, "kermany_ped": ker, "shenzhen": shz}.items()}

    rows = []
    for sc in ("mahalanobis_l2", "knn_mean_cosine"):
        s = get_scorer(sc).fit(F["fit"])
        S = {k: np.asarray(s.score(F[k]), dtype=np.float64) for k in ("seen", "unseen", "kermany_ped", "shenzhen")}
        for ood in ("kermany_ped", "shenzhen"):
            for idv in ("seen", "unseen"):
                y = np.r_[np.ones(len(S[idv])), np.zeros(len(S[ood]))]
                rows.append({"scorer": sc, "ood": ood, "id": idv, "n_id": len(S[idv]), "n_ood": len(S[ood]),
                             "auroc": float(roc_auc_score(y, np.r_[S[idv], S[ood]]))})

    def maha_max(ood):
        return max(r["auroc"] for r in rows if r["scorer"] == "mahalanobis_l2" and r["ood"] == ood)

    m = {o: maha_max(o) for o in ("kermany_ped", "shenzhen")}
    order = ["kermany_ped", "shenzhen"]
    ok = [o for o in order if m[o] <= 0.97]
    if ok:
        primary, rule = ok[0], "first non-saturating candidate"
    else:
        primary, rule = min(order, key=lambda o: m[o]), "both saturate: lower max Maha AUROC"
    secondary = [o for o in order if o != primary][0]

    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUT / "pilot_ood.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"info": info, "maha_max_auroc": m, "saturation_threshold": 0.97, "primary": primary,
           "secondary": secondary, "rule_applied": rule, "seconds": round(time.time() - t0, 1)}
    (OUT / "pilot_ood.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
