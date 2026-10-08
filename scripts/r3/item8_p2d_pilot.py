#!/usr/bin/env python3
"""R3 item 8 / P2-d OOD ceiling pilot, as fixed in results/r3/8/p2d/pilot_plan.md (CPU, frozen DINOv2-B).

No same-group vs other-group fits: OOD AUROC only, never Delta. Candidates: Kvasir-Capsule Angiectasia, Erosion
(K_seg roles); brain meningioma, pituitary, glioma (record-wise roles). Images from the staged 256 px tars.
Output: results/r3/8/p2d/pilot_ood.csv, results/r3/8/p2d/pilot_ood.json
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
P2D, STG = REPO / "results/r3/8/p2d", REPO / "data/staged"
N_EVAL, N_OOD, FIT_CAP, THR = 300, 300, 1500, 0.97
ORDER = {"kvasir_capsule": ["Angiectasia", "Erosion"], "brain": ["meningioma", "pituitary", "glioma"]}


def sample(idx, n, seed):
    idx = np.asarray(sorted(idx))
    return idx if len(idx) <= n else np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))


def sets(t):
    """t: rows of one candidate with columns role, grp, member; returns member lists and counts."""
    t = t.reset_index(drop=True)
    fit = sample(np.flatnonzero(t.role == "fit"), FIT_CAP, 3)
    fit_g = set(t.grp[fit])
    seen = sample(np.flatnonzero((t.role == "seen") & t.grp.isin(fit_g)), N_EVAL, 1)
    unseen = sample(np.flatnonzero(t.role == "unseen"), N_EVAL, 2)
    ood = sample(np.flatnonzero(t.role == "ood"), N_OOD, 4)
    S = {"fit": fit, "seen": seen, "unseen": unseen, "ood": ood}
    info = {k: {"n": len(v), "groups": int(t.grp[v].nunique())} for k, v in S.items()}
    info["seen_eligible"] = int(((t.role == "seen") & t.grp.isin(fit_g)).sum())
    return {k: t.member[v].tolist() for k, v in S.items()}, info


def tables():
    k = pd.read_csv(P2D / "kvasir_sets.csv.gz", dtype={"file": str, "video": str})
    im = pd.read_csv(P2D / "kvasir_images.csv.gz", dtype={"file": str}).drop_duplicates("file").set_index("file").folder
    k["role"], k["grp"] = k.role_seg, k.video
    k["member"] = list(zip(im.loc[k.file].values, k.file + ".png"))
    b = pd.read_csv(P2D / "brain_sets.csv.gz")
    b["grp"], b["member"] = b.pid.astype(str), [("brain", f"{i}.png") for i in b.idx]
    return {"kvasir_capsule": (k, "p2d_kvasir_256"), "brain": (b, "p2d_brain_256")}


def read_members(stage, members):
    from PIL import Image
    want = {}
    for tar_name, m in members:
        want.setdefault(tar_name, set()).add(m)
    out = {}
    for tar_name, ms in want.items():
        with tarfile.open(STG / stage / f"{tar_name}.tar") as t:
            for ti in t:
                if ti.name in ms:
                    out[(tar_name, ti.name)] = Image.open(io.BytesIO(t.extractfile(ti).read())).convert("RGB")
    missing = set(members) - set(out)
    assert not missing, list(missing)[:5]
    return out


def embed(imgs, model, tf, bs=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.stack([tf(im) for im in imgs[i:i + bs]])
            out.append(model.forward_features(x)["x_norm_clstoken"].float().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    import torch
    from torchvision import transforms as T
    from crossfit_ood import get_scorer

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    t0 = time.time()
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=True).eval()
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    rows, res = [], {}
    for ds, (tab, stage) in tables().items():
        cand = {}
        for c in ORDER[ds]:
            S, info = sets(tab[tab.ood_candidate == c])
            imgs = read_members(stage, sorted({m for v in S.values() for m in v}))
            cache = {}
            F = {}
            for k, v in S.items():
                todo = [m for m in v if m not in cache]
                if todo:
                    cache.update(zip(todo, embed([imgs[m] for m in todo], model, tf)))
                F[k] = np.stack([cache[m] for m in v])
            for sc in ("mahalanobis_l2", "knn_mean_cosine"):
                s = get_scorer(sc).fit(F["fit"])
                Sc = {k: np.asarray(s.score(F[k]), dtype=np.float64) for k in ("seen", "unseen", "ood")}
                for idv in ("seen", "unseen"):
                    y = np.r_[np.ones(len(Sc[idv])), np.zeros(len(Sc["ood"]))]
                    rows.append({"dataset": ds, "candidate": c, "scorer": sc, "id": idv, "n_id": len(Sc[idv]),
                                 "n_ood": len(Sc["ood"]), "auroc": float(roc_auc_score(y, np.r_[Sc[idv], Sc["ood"]]))})
            mx = max(r["auroc"] for r in rows if r["dataset"] == ds and r["candidate"] == c and r["scorer"] == "mahalanobis_l2")
            cand[c] = {"sets": info, "maha_max_auroc": mx, "skipped": bool(mx > THR)}
            print(ds, c, info, round(mx, 4), round(time.time() - t0), flush=True)
        ok = [c for c in ORDER[ds] if not cand[c]["skipped"]]
        if ds == "brain" and not ok:
            primary, secondary, rule = None, None, "all three skipped: brain P2-d 'ceiling - not identifiable', stops"
        elif ok:
            primary, rule = ok[0], "first non-skipped candidate"
            secondary = ok[1] if len(ok) > 1 else None
        else:
            primary, secondary, rule = None, None, "Angiectasia and Erosion skipped: Galar candidate reached (addendum before running)"
        res[ds] = {"candidates": cand, "primary": primary, "secondary": secondary, "rule_applied": rule}
    with open(P2D / "pilot_ood.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = {"plan": "results/r3/8/p2d/pilot_plan.md", "saturation_threshold": THR, **res,
           "seconds": round(time.time() - t0, 1)}
    (P2D / "pilot_ood.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
