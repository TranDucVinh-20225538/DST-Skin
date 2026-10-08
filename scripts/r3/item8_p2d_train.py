#!/usr/bin/env python3
"""R3 item 8 / P2-d CNN training + feature extraction for one (dataset, variant, arch, seed) (results/r3/8/p2d/PRECOMMIT.json).

Recipe = P2-a / medbench L2: cnn_family.recipe(arch), ImageNet init, skin_newcnn train / eval transforms at 224, cosine
(T_max = epochs), set_seed(seed), epochs = clip(ceil(300000 / n_fit), 10, 50). Task = single-label classification over the
ID classes of the primary candidate (cross-entropy). Training set = role fit of the variant; checkpoint = best macro
one-vs-rest AUROC on role ckpt. Features (penultimate, extract_features_and_logits) + logits of the best checkpoint for
fit / seen / unseen / ood. Images: staged 256 px PNGs copied to node-local --img-dir ({key}.png).
Writes data/models/r3_item8/p2d/{tag}_best.pth (not committed), outputs/rigor_pack/r3_item8/p2d/{tag}.npz,
results/r3/8/p2d/train/{tag}.json with tag = {ds}_{variant}_{arch}_s{seed}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/r3")]
from item8_p2d_common import FEAT, P2D, macro_ovr_auroc, variant_sets  # noqa: E402


class Imgs:
    def __init__(self, paths, labels, tfm):
        self.p, self.y, self.t = list(paths), np.asarray(labels, dtype=np.int64), tfm

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            x = self.t(im.convert("RGB"))
        return x, self.y[i], i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["kvasir", "brain"])
    ap.add_argument("--variant", required=True, choices=["lit", "seg", "rec"])
    ap.add_argument("--arch", required=True, choices=["resnet50", "convnext_tiny"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
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

    V = variant_sets(args.ds, args.variant)
    if V is None:
        print("no primary candidate for", args.ds, "- nothing to train")
        return 0
    a, s = args.arch, args.seed
    smoke = args.max_epochs is not None or args.max_n is not None
    tag = f"{args.ds}_{args.variant}_{a}_s{s}" + ("_smoke" if smoke else "")
    classes, img = V["classes"], Path(args.img_dir)
    cidx = {c: i for i, c in enumerate(classes)}
    sets = {}
    for r in ("fit", "ckpt", "seen", "unseen", "ood"):
        x = V["sets"][r]
        if args.max_n:
            x = x.sample(n=min(args.max_n, len(x)), random_state=0) if r == "fit" else x.iloc[: args.max_n]
        y = np.array([cidx.get(l, -1) for l in x.label])
        sets[r] = ([img / f"{k}.png" for k in x.key], y, x.key.to_numpy(), x.label.to_numpy())
    n_fit = len(V["sets"]["fit"])
    E = int(min(50, max(10, math.ceil(300000 / n_fit)))) if args.max_epochs is None else args.max_epochs
    size = 224
    tf_tr, tf_ev = build_train_transform(a, size), build_eval_transform(a, size)
    rec = recipe(a)
    bs = int(rec["batch_size"])
    dev = torch.device("cuda")
    mdir, rdir = REPO / "data/models/r3_item8/p2d", P2D / "train"
    for p in (mdir, FEAT, rdir):
        p.mkdir(parents=True, exist_ok=True)
    best_p = mdir / f"{tag}_best.pth"
    dl = lambda k, t, sh: DataLoader(Imgs(sets[k][0], sets[k][1], t), batch_size=bs, shuffle=sh,  # noqa: E731
                                     num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"{tag}: candidate={V['candidate']} classes={classes} " + ", ".join(f"{k}={len(v[0])}" for k, v in sets.items())
          + f"; epochs={E} bs={bs}", flush=True)

    model = get_cnn_backbone(a, num_classes=len(classes), pretrained=True).to(dev)
    t0 = time.time()
    set_seed(s)
    tr_l, va_l = dl("fit", tf_tr, True), dl("ckpt", tf_ev, False)
    opt = make_optimizer(model, a)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    ce = nn.CrossEntropyLoss()
    hist, best_auc, best_ep = [], -1.0, 0
    for ep in range(1, E + 1):
        te = time.time()
        model.train()
        run, n = 0.0, 0
        for x, y, _ in tr_l:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        model.eval()
        P = []
        with torch.no_grad():
            for x, _, _ in va_l:
                P.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
        auc, _ = macro_ovr_auroc(sets["ckpt"][3], np.concatenate(P), classes)
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
    for k in ("fit", "seen", "unseen", "ood"):
        lo, fe = extract_features_and_logits(model, dl(k, tf_ev, False), dev)
        out[f"{k}_logits"], out[f"{k}_feats"] = lo.astype(np.float32), fe.astype(np.float32)
        out[f"{k}_keys"] = sets[k][2]
        if k != "ood":
            lo64 = lo.astype(np.float64)
            p = np.exp(lo64 - lo64.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            ok = np.ones(len(p), bool)
            if k == "seen":
                g = V["sets"]["seen"].set_index("key").grp.loc[sets[k][2]].to_numpy()
                ok = np.isin(g, V["sets"]["fit"].grp.unique())
            m, per = macro_ovr_auroc(sets[k][3][ok], p[ok], classes)
            res_auc[k] = {"macro": m, "per_class": per, "acc": float((p[ok].argmax(1) == sets[k][1][ok]).mean()),
                          "n": int(ok.sum())}
    np.savez(FEAT / f"{tag}.npz", **out)
    t_ext = time.time() - t1
    g_loss = hist[-1]["loss"] <= 0.5 * hist[0]["loss"]
    g_auc = res_auc["seen"]["macro"] >= 0.75
    res = {"ds": args.ds, "variant": args.variant, "candidate": V["candidate"], "arch": a, "seed": s, "smoke": smoke,
           "classes": classes, "epochs": E, "best_epoch": best_ep, "best_ckpt_macro_auroc": best_auc,
           "n": {k: len(v[0]) for k, v in sets.items()}, "d": int(out["fit_feats"].shape[1]), "recipe": rec,
           "input_size": size, "loss": "CrossEntropy (single-label)", "epoch1_loss": hist[0]["loss"],
           "final_loss": hist[-1]["loss"], "id_macro_auroc": {k: v["macro"] for k, v in res_auc.items()},
           "id_acc": {k: v["acc"] for k, v in res_auc.items()},
           "id_per_class_auroc": {k: v["per_class"] for k, v in res_auc.items()},
           "G_competence": {"final_loss_le_half_epoch1": bool(g_loss), "seen_macro_auroc_ge_0.75": bool(g_auc),
                            "pass": bool(g_loss and g_auc)},
           "train_seconds": round(t_train, 1), "extract_seconds": round(t_ext, 1), "gpu": torch.cuda.get_device_name(0),
           "history": hist}
    (rdir / f"{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "id_per_class_auroc")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
