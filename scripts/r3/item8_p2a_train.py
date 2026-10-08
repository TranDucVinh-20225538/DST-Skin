#!/usr/bin/env python3
"""R3 item 8 / P2-a CNN training + feature extraction for one (arch, seed) (results/r3/8/PRECOMMIT.json).

Recipe = medbench L2 (scripts/rigor/medbench_train.py): cnn_family.recipe(arch), ImageNet init, skin_newcnn
train / eval transforms at 224, cosine (T_max = epochs), set_seed(seed), epochs = clip(ceil(300000 / n_fit), 10, 50).
Task = 14-finding multi-label classification (BCE with logits), as precommitted. Training set = role 'fit' of
results/r3/8/p2a/sets.csv.gz; checkpoint = best macro AUROC (14 findings) on role 'ckpt' (10% of the training
patients, patient-disjoint from fit).
Features (penultimate, extract_features_and_logits) + logits of the best checkpoint for fit / seen / unseen;
--ood-only: the same for OOD (Shenzhen + Kermany pediatric, order of results/r3/8/p2a/ood.csv.gz) from the saved checkpoint.
Images: NIH staged 256 PNGs (--img-dir), OOD staged 256 PNGs (--ood-dir), both node-local copies.
Writes data/models/r3_item8/p2a/{arch}_s{seed}_best.pth (not committed), outputs/rigor_pack/r3_item8/p2a/{arch}_s{seed}.npz
(+ {arch}_s{seed}_ood.npz), results/r3/8/p2a/train/{arch}_s{seed}.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
P2A = REPO / "results/r3/8/p2a"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


class Imgs:
    def __init__(self, paths, labels, tfm):
        self.p, self.y, self.t = list(paths), np.asarray(labels, dtype=np.float32), tfm

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            x = self.t(im.convert("RGB"))
        return x, self.y[i], i


def macro_auroc(Y, P):
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, f in enumerate(FINDINGS):
        if 0 < Y[:, j].sum() < len(Y):
            per[f] = float(roc_auc_score(Y[:, j], P[:, j]))
    return float(np.mean(list(per.values()))), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["resnet50", "convnext_tiny"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ood-dir", default=None)
    ap.add_argument("--ood-only", action="store_true", help="extract OOD features from the saved best checkpoint")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--max-n", type=int, default=None, help="smoke only: cap images per set")
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    from scripts.skin_newcnn import build_eval_transform, build_train_transform, make_optimizer, set_seed
    from src.models.cnn_family import get_cnn_backbone, recipe
    from src.utils.feature_extractor import extract_features_and_logits

    a, s = args.arch, args.seed
    smoke = args.max_epochs is not None or args.max_n is not None
    tag = f"{a}_s{s}" + ("_smoke" if smoke else "")
    S = pd.read_csv(P2A / "sets.csv.gz")
    img = Path(args.img_dir)
    sets = {}
    for r in ("fit", "ckpt", "seen", "unseen"):
        x = S[S.role == r]
        if args.max_n:
            x = x.iloc[: args.max_n]
        sets[r] = ([img / i for i in x.image], x[FINDINGS].to_numpy(), x.image.to_numpy())
    if args.ood_only:
        O = pd.read_csv(P2A / "ood.csv.gz")
        xo = O.iloc[: args.max_n] if args.max_n else O
        sets["ood"] = ([Path(args.ood_dir) / (k + ".png") for k in xo.key], np.zeros((len(xo), len(FINDINGS))),
                       xo.key.to_numpy())
    n_fit = int((S.role == "fit").sum())
    E = int(min(50, max(10, math.ceil(300000 / n_fit)))) if args.max_epochs is None else args.max_epochs
    size = 224
    tf_tr, tf_ev = build_train_transform(a, size), build_eval_transform(a, size)
    rec = recipe(a)
    bs = int(rec["batch_size"])
    dev = torch.device("cuda")
    mdir, fdir, rdir = REPO / "data/models/r3_item8/p2a", REPO / "outputs/rigor_pack/r3_item8/p2a", P2A / "train"
    for p in (mdir, fdir, rdir):
        p.mkdir(parents=True, exist_ok=True)
    best_p = mdir / f"{tag}_best.pth"
    dl = lambda k, t, sh: DataLoader(Imgs(sets[k][0], sets[k][1], t), batch_size=bs, shuffle=sh,  # noqa: E731
                                     num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"{tag}: " + ", ".join(f"{k}={len(v[0])}" for k, v in sets.items()) + f"; epochs={E} bs={bs}", flush=True)

    model = get_cnn_backbone(a, num_classes=len(FINDINGS), pretrained=True).to(dev)
    if args.ood_only:
        model.load_state_dict(torch.load(best_p, map_location=dev))
        lo, fe = extract_features_and_logits(model, dl("ood", tf_ev, False), dev)
        np.savez(fdir / f"{tag}_ood.npz", ood_logits=lo.astype(np.float32), ood_feats=fe.astype(np.float32),
                 ood_keys=sets["ood"][2])
        print(tag, "ood", fe.shape, flush=True)
        return 0

    t0 = time.time()
    set_seed(s)
    tr_l, va_l = dl("fit", tf_tr, True), dl("ckpt", tf_ev, False)
    opt = make_optimizer(model, a)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    bce = nn.BCEWithLogitsLoss()
    hist, best_auc, best_ep = [], -1.0, 0
    for ep in range(1, E + 1):
        te = time.time()
        model.train()
        run, n = 0.0, 0
        for x, y, _ in tr_l:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = bce(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y, _ in va_l:
                P.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
                Y.append(y.numpy())
        auc, _ = macro_auroc(np.concatenate(Y), np.concatenate(P))
        hist.append({"epoch": ep, "loss": run / n, "ckpt_macro_auroc": auc, "seconds": time.time() - te})
        print("[Epoch %02d/%02d] loss=%.5f ckpt_macro_auroc=%.4f %.0fs" % (ep, E, run / n, auc, hist[-1]["seconds"]),
              flush=True)
        if auc > best_auc:
            best_auc, best_ep = auc, ep
            torch.save(model.state_dict(), best_p)
    t_train = time.time() - t0

    t1 = time.time()
    model.load_state_dict(torch.load(best_p, map_location=dev))
    out, res_auc = {}, {}
    for k in ("fit", "seen", "unseen"):
        lo, fe = extract_features_and_logits(model, dl(k, tf_ev, False), dev)
        out[f"{k}_logits"], out[f"{k}_feats"] = lo.astype(np.float32), fe.astype(np.float32)
        out[f"{k}_keys"] = sets[k][2]
        out[f"{k}_labels"] = sets[k][1].astype(np.int8)
        m, per = macro_auroc(sets[k][1], 1 / (1 + np.exp(-lo.astype(np.float64))))
        res_auc[k] = {"macro": m, "per_finding": per}
    np.savez(fdir / f"{tag}.npz", **out)
    t_ext = time.time() - t1
    g_loss = hist[-1]["loss"] <= 0.5 * hist[0]["loss"]
    g_auc = res_auc["seen"]["macro"] >= 0.75
    res = {"arch": a, "seed": s, "smoke": smoke, "epochs": E, "best_epoch": best_ep, "best_ckpt_macro_auroc": best_auc,
           "n": {k: len(v[0]) for k, v in sets.items()}, "d": int(out["fit_feats"].shape[1]), "recipe": rec,
           "input_size": size, "loss": "BCEWithLogits (14 findings)", "epoch1_loss": hist[0]["loss"],
           "final_loss": hist[-1]["loss"], "id_macro_auroc": {k: v["macro"] for k, v in res_auc.items()},
           "id_per_finding_auroc": {k: v["per_finding"] for k, v in res_auc.items()},
           "G_competence": {"final_loss_le_half_epoch1": bool(g_loss), "seen_macro_auroc_ge_0.75": bool(g_auc),
                            "pass": bool(g_loss and g_auc)},
           "train_seconds": round(t_train, 1), "extract_seconds": round(t_ext, 1), "history": hist}
    (rdir / f"{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "id_per_finding_auroc")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
