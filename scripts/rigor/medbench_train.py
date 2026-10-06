#!/usr/bin/env python3
"""Medbench training + feature extraction for one (dataset, arch, seed, arm) run
(decisions/precommit_group_leakage_medbench_2026-10-06.md, L2/L3/L5).

Recipe: cnn_family.recipe(arch), ImageNet init, cosine (T_max = epochs_d), the train / eval transforms
of skin_newcnn.py at the L2 input size, set_seed(seed). Checkpoint = best macro one-vs-rest val AUC.
Images are read from the node-local copy of data/staged/<ds>_<size>.tar (--img-dir).
Writes data/models/medbench/<ds>/seed<s>/<arch>_<arm>_best.pth (not committed),
outputs/rigor_pack/medbench/<ds>/<arch>_s<s>_<arm>.npz (feats / logits / labels / keys per set),
outputs/reports/rigor_pack/medbench/<ds>/train_<arch>_s<s>_<arm>.json (history, gate inputs).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import medbench_common as MC  # noqa: E402

REPO = MC.REPO


class Imgs:
    def __init__(self, img_dir: Path, sel: dict, tfm):
        self.p = [img_dir / MC.member(k) for k in sel["keys"]]
        self.y = sel["labels"]
        self.t = tfm

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            x = self.t(im.convert("RGB"))
        return x, int(self.y[i]), i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(MC.CLASSES))
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke only")
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from scripts.skin_newcnn import build_eval_transform, build_train_transform, make_optimizer, set_seed
    from src.models.cnn_family import fc_params, get_cnn_backbone, recipe
    from src.utils.feature_extractor import extract_features_and_logits

    ds, a, s, armn = args.ds, args.arch, args.seed, args.arm
    E = MC.EPOCHS[ds] if args.max_epochs is None else args.max_epochs
    tag = f"{a}_s{s}_{armn}" + ("_smoke" if args.max_epochs else "")
    sets = MC.arm(ds, armn)
    ncls = len(MC.CLASSES[ds])
    img_dir = Path(args.img_dir)
    size = MC.input_size(a)
    tf_tr, tf_ev = build_train_transform(a, size), build_eval_transform(a, size)
    rec = recipe(a)
    bs = int(rec["batch_size"])
    dev = torch.device("cuda")
    mdir = REPO / f"data/models/medbench/{ds}/seed{s}"
    fdir = REPO / f"outputs/rigor_pack/medbench/{ds}"
    rdir = REPO / f"outputs/reports/rigor_pack/medbench/{ds}"
    for p in (mdir, fdir, rdir):
        p.mkdir(parents=True, exist_ok=True)
    best_p = mdir / f"{tag}_best.pth"
    dl = lambda sel, t, sh: DataLoader(Imgs(img_dir, sel, t), batch_size=bs, shuffle=sh, num_workers=args.workers,  # noqa: E731
                                       pin_memory=True, persistent_workers=False, drop_last=False)
    print(f"{ds} {tag}: " + ", ".join(f"{k}={len(v['keys'])}" for k, v in sets.items()) + f"; epochs={E} bs={bs}", flush=True)

    t0 = time.time()
    set_seed(s)
    tr_l, va_l = dl(sets["train"], tf_tr, True), dl(sets["val"], tf_ev, False)
    model = get_cnn_backbone(a, num_classes=ncls, pretrained=True).to(dev)
    opt = make_optimizer(model, a)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    ce = nn.CrossEntropyLoss()
    hist, best_auc, best_ep = [], -1.0, 0
    for ep in range(1, E + 1):
        te = time.time()
        model.train()
        run, n, cor = 0.0, 0, 0
        for x, y, _ in tr_l:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y)
            loss.backward()
            opt.step()
            run += loss.item() * y.size(0)
            cor += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        sched.step()
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y, _ in va_l:
                P.append(torch.softmax(model(x.to(dev)), 1).cpu().numpy())
                Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        present = np.unique(Y)
        auc = float(roc_auc_score(Y, P[:, present] / P[:, present].sum(1, keepdims=True), multi_class="ovr",
                                  average="macro", labels=present)) if len(present) > 2 else \
            float(roc_auc_score(Y == present[-1], P[:, present[-1]]))
        vacc = float((P.argmax(1) == Y).mean())
        hist.append({"epoch": ep, "loss": run / n, "train_acc": cor / n, "val_auc": auc, "val_acc": vacc,
                     "seconds": time.time() - te})
        print("[Epoch %02d/%02d] loss=%.4f train_acc=%.4f val_auc=%.4f val_acc=%.4f %.0fs" % (
            ep, E, run / n, cor / n, auc, vacc, hist[-1]["seconds"]), flush=True)
        if auc > best_auc:
            best_auc, best_ep = auc, ep
            torch.save(model.state_dict(), best_p)
    t_train = time.time() - t0

    t1 = time.time()
    model.load_state_dict(torch.load(best_p, map_location=dev))
    w, b = fc_params(model)
    out = {"fc_weight": np.asarray(w), "fc_bias": np.asarray(b)}
    for k, sel in sets.items():
        if k == "val":
            continue
        lo, fe = extract_features_and_logits(model, dl(sel, tf_ev, False), dev)
        out[f"{k}_logits"], out[f"{k}_feats"] = lo.astype(np.float32), fe.astype(np.float32)
        out[f"{k}_labels"], out[f"{k}_keys"] = sel["labels"], np.array(sel["keys"])
    np.savez(fdir / f"{tag}.npz", **out)
    t_ext = time.time() - t1
    res = {"ds": ds, "arch": a, "seed": s, "arm": armn, "epochs": E, "best_epoch": best_ep, "best_val_auc": best_auc,
           "n": {k: len(v["keys"]) for k, v in sets.items()}, "recipe": rec, "input_size": size,
           "final_train_acc": hist[-1]["train_acc"], "final_loss": hist[-1]["loss"], "epoch1_loss": hist[0]["loss"],
           "train_seconds": round(t_train, 1), "extract_seconds": round(t_ext, 1), "history": hist}
    for k in sets:
        if k != "val" and not k.startswith("ood") and f"{k}_logits" in out:
            res[f"acc_{k}"] = float((out[f"{k}_logits"].argmax(1) == out[f"{k}_labels"]).mean())
    (rdir / f"train_{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "history"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
