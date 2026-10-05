#!/usr/bin/env python3
"""Slide-disjoint retrain on balanced folds, seeds 42-44 (decisions/precommit_isbi_patch2_2026-10-06.md, P2/P3).

One (arch, seed, fold) per call. Training exactly as logit_retrain_slide_disjoint.py (patch 1) except
the folds (balanced_folds below) and set_seed(seed); best checkpoint on the same fold's id_val
slides, last-epoch checkpoint also saved. Then avgpool features / logits (eval transform) of the
fold's train patches, all id_val and all hospital 2 for the P5 feature scores, and MSP / Energy
(T=1) AUROC on unseen-slide (other fold) vs seen-slide (same fold) id_val.

Checkpoints: data/models/camelyon17/frac1/slide_disjoint_v2/seed{s}/{arch}_fold{f}_{best,last}.pth
Features:    outputs/rigor_pack/logit_retrain_slide_disjoint_v2/{arch}_s{s}_f{f}.npz
Result:      outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint_v2/{arch}_s{s}_f{f}.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from logit_retrain_slide_disjoint import load_pilot  # noqa: E402

FOLD_RNG = 20261006


def balanced_folds(tsl, thosp, tr_patches_per_slide, vsl):
    """5 + 5 slides per hospital minimising the train-patch difference; ties by default_rng(20261006)."""
    rng = np.random.default_rng(FOLD_RNG)
    fold_of = {}
    for h in sorted(np.unique(thosp)):
        sl = sorted(np.unique(tsl[thosp == h]).tolist())
        tot = sum(tr_patches_per_slide[x] for x in sl)
        cands = []
        for a in itertools.combinations(sl, len(sl) // 2):
            n0 = sum(tr_patches_per_slide[x] for x in a)
            cands.append((abs(2 * n0 - tot), a))
        best = min(c[0] for c in cands)
        ties = [a for d, a in cands if d == best]
        pick = ties[int(rng.integers(len(ties)))]
        for x in sl:
            fold_of[x] = 0 if x in pick else 1
    return np.array([fold_of[x] for x in tsl]), np.array([fold_of.get(x, -1) for x in vsl])


def fold_assignment(meta):
    sp = meta.wilds_split.to_numpy()
    tr_idx, iv_idx = np.where(sp == 0)[0], np.where(sp == 1)[0]
    tsl, vsl = meta.slide.to_numpy()[tr_idx], meta.slide.to_numpy()[iv_idx]
    cnt = dict(zip(*np.unique(tsl, return_counts=True)))
    tf, vf = balanced_folds(tsl, meta.center.to_numpy()[tr_idx], cnt, vsl)
    return tr_idx, iv_idx, tf, vf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["resnet50", "convnext_tiny", "densenet121"])
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1])
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from wilds.datasets.wilds_dataset import WILDSSubset

    from src.datasets.camelyon_ood import _WildsPathDataset, build_transform, get_wilds_dataset
    from src.models.cnn_family import fc_params
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.feature_extractor import extract_features_and_logits
    from src.utils.scoring import OODScorer

    P = load_pilot()
    a, s, f = args.arch, args.seed, args.fold
    mdir = C.ensure_dir(C.REPO / ("data/models/camelyon17/frac1/slide_disjoint_v2/seed%d" % s))
    fdir = C.ensure_dir(C.default_out(C.REPO) / "logit_retrain_slide_disjoint_v2")
    rdir = C.ensure_dir(C.default_reports(C.REPO) / "leakage/logit_retrain_slide_disjoint_v2")
    best, last = mdir / ("%s_fold%d_best.pth" % (a, f)), mdir / ("%s_fold%d_last.pth" % (a, f))
    meta_p = mdir / ("%s_fold%d_train_meta.json" % (a, f))

    meta = C.load_camelyon_metadata(C.REPO)
    tr_idx, iv_idx, tf, vf = fold_assignment(meta)
    sizes = (int((tf == 0).sum()), int((tf == 1).sum()))
    print("balanced folds: train %s, id_val %s" % (sizes, ((vf == 0).sum(), (vf == 1).sum())), flush=True)

    P.wait_for_camelyon()
    ds = get_wilds_dataset(download=False)
    if not np.array_equal(np.where(ds.split_array == ds.split_dict["train"])[0], tr_idx):
        raise SystemExit("STOP: WILDS train indices differ from metadata.csv train rows")
    rec = P.resolve_recipe(a)
    bs = int(rec["batch_size"])
    train_tf, eval_tf = build_transform(a, train=True), build_transform(a, train=False)
    mk = lambda idx, tfm, name, sh: DataLoader(_WildsPathDataset(WILDSSubset(ds, idx, tfm), name), batch_size=bs,  # noqa: E731
                                               shuffle=sh, num_workers=args.num_workers, pin_memory=True)
    device = P.get_device()
    t0 = time.time()
    if not meta_p.exists():
        P.set_seed(s)
        train_l = mk(tr_idx[tf == f], train_tf, "camelyon17/train", True)
        sel_l = mk(iv_idx[vf == f], eval_tf, "camelyon17/id_val", False)
        model = P.get_model(a).to(device)
        crit = nn.CrossEntropyLoss()
        opt = P.make_optimizer(model, a, rec=rec)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.NUM_EPOCHS)
        n_train = len(train_l.dataset)
        print("=== v2 retrain %s seed %d fold %d n_train=%d n_sel=%d optim=%s lr=%s wd=%s bs=%d ===" % (
            a, s, f, n_train, len(sel_l.dataset), rec["optim"], rec["lr"], rec["wd"], bs), flush=True)
        best_acc, best_ep, hist = -1.0, 0, []
        for ep in range(1, P.NUM_EPOCHS + 1):
            te = time.time()
            model.train()
            run = 0.0
            for x, y, _ in train_l:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
                run += loss.item() * y.size(0)
            sched.step()
            acc, _ = P.evaluate(model, sel_l, device)
            hist.append({"epoch": ep, "loss": run / n_train, "sel_acc": acc, "seconds": time.time() - te})
            print("[Epoch %02d/%02d] loss=%.4f sel_acc=%.4f %.0fs" % (ep, P.NUM_EPOCHS, run / n_train, acc, hist[-1]["seconds"]), flush=True)
            if acc > best_acc:
                best_acc, best_ep = acc, ep
                torch.save(model.state_dict(), best)
        torch.save(model.state_dict(), last)
        meta_p.write_text(json.dumps({"arch": a, "fold": f, "seed": s, "n_train": n_train, "best_sel_acc": best_acc,
                                      "best_epoch": best_ep, "optim": rec["optim"], "lr": rec["lr"], "wd": rec["wd"],
                                      "batch_size": bs, "history": hist}, indent=2) + "\n")
        del model
    t_train = time.time() - t0
    tm = json.loads(meta_p.read_text())

    t1 = time.time()
    model = P.get_model(a, pretrained=False)
    model.load_state_dict(torch.load(best, map_location=device))
    model.to(device)
    oo = DataLoader(_WildsPathDataset(ds.get_subset("test", transform=eval_tf), "camelyon17/test"), batch_size=bs,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
    tr_lo, tr_fe = extract_features_and_logits(model, mk(tr_idx[tf == f], eval_tf, "camelyon17/train", False), device)
    iv_lo, iv_fe = extract_features_and_logits(model, mk(iv_idx, eval_tf, "camelyon17/id_val", False), device)
    oo_lo, oo_fe = extract_features_and_logits(model, oo, device)
    w, b = fc_params(model)
    np.savez(fdir / ("%s_s%d_f%d.npz" % (a, s, f)), train_feats=tr_fe, train_logits=tr_lo, val_feats=iv_fe,
             val_logits=iv_lo, ood_feats=oo_fe, ood_logits=oo_lo, fc_weight=np.asarray(w.detach().cpu()),
             fc_bias=np.asarray(b.detach().cpu()), id_val_fold=vf)
    model.load_state_dict(torch.load(last, map_location=device))
    model.to(device)
    iv_last = P.collect_logits(model, mk(iv_idx, eval_tf, "camelyon17/id_val", False), device)
    oo_last = P.collect_logits(model, oo, device)
    t_extract = time.time() - t1

    yi, yo = meta.tumor.to_numpy()[iv_idx], meta.tumor.to_numpy()[meta.wilds_split.to_numpy() == 2]
    d, sm = vf == 1 - f, vf == f
    out = {"arch": a, "seed": s, "fold": f, "n_train": sizes[f], "n_id_unseen": int(d.sum()), "n_id_seen": int(sm.sum()),
           "n_ood": len(oo_lo), "train_tumour_frac": float(meta.tumor.to_numpy()[tr_idx][tf == f].mean()),
           "best_epoch": tm["best_epoch"], "best_sel_acc": tm["best_sel_acc"],
           "acc_id_unseen": float((iv_lo[d].argmax(1) == yi[d]).mean()),
           "acc_id_seen": float((iv_lo[sm].argmax(1) == yi[sm]).mean()),
           "acc_ood": float((oo_lo.argmax(1) == yo).mean()),
           "pred_tumour_ood": float(oo_lo.argmax(1).mean()),
           "train_seconds": round(t_train, 1), "extract_seconds": round(t_extract, 1),
           "epoch_seconds": [h["seconds"] for h in tm["history"]]}
    for name, fn in (("msp", OODScorer.score_msp), ("energy", lambda z: OODScorer.score_energy(z, 1.0))):
        for tag, I, O in (("", iv_lo, oo_lo), ("_last", iv_last, oo_last)):
            out["auroc_%s_unseen%s" % (name, tag)] = calc_auroc(fn(I[d]), fn(O))
            out["auroc_%s_seen%s" % (name, tag)] = calc_auroc(fn(I[sm]), fn(O))
        out["gap_%s" % name] = out["auroc_%s_seen" % name] - out["auroc_%s_unseen" % name]
    out["seconds_total"] = round(time.time() - t0, 1)
    (rdir / ("%s_s%d_f%d.json" % (a, s, f))).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
