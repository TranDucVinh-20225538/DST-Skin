#!/usr/bin/env python3
"""Multibench training (decisions/precommit_group_leakage_multibench_2026-10-06.md).

--fold 0/1: (B) retrain on that fold's train groups; then features of the fold's train images, all ID
            and all OOD (for the same-protocol scores) and MSP / Energy AUROC on seen-group vs
            unseen-group ID (within-model gap).
--fold full: RxRx1 (A) base model on the full train split (seed 42), then indexed features.
Recipe: cnn_recipe(arch), 10 epochs, cosine, ImageNet init, set_seed(seed). iWildCam: transforms of
iwildcam_ood, checkpoint by accuracy on the same fold's id_val locations (">=", as iwildcam_pilot).
RxRx1: transforms of multibench_common, last-epoch checkpoint.

Checkpoints: data/models/multibench/{ds}/seed{s}/{arch}_{fold}_{best,last}.pth
Features:    outputs/rigor_pack/multibench/{ds}/{arch}_s{s}_f{fold}.npz  (full: {arch}_s42_full.npz)
Result:      outputs/reports/rigor_pack/multibench/{ds}/retrain_{arch}_s{s}_f{fold}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import multibench_common as M  # noqa: E402
from multibench_extract import extract  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(M.SPECS))
    ap.add_argument("--arch", required=True, choices=list(M.ARCHS_B))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--fold", required=True, choices=["0", "1", "full"])
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from scripts.camelyon17_pilot import NUM_EPOCHS, evaluate, make_optimizer, set_seed
    from src.models.cnn_family import fc_params, get_cnn_backbone, recipe
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.feature_extractor import extract_features_and_logits
    from src.utils.scoring import OODScorer

    ds, a, s, fl = args.ds, args.arch, args.seed, args.fold
    sp = M.SPECS[ds]
    if fl == "full" and (ds != "rxrx1" or s != 42):
        raise SystemExit("full training is only the RxRx1 seed-42 base model")
    mdir = C.ensure_dir(C.REPO / ("data/models/multibench/%s/seed%d" % (ds, s)))
    best, last = mdir / ("%s_%s_best.pth" % (a, fl)), mdir / ("%s_%s_last.pth" % (a, fl))
    meta_p = mdir / ("%s_%s_train_meta.json" % (a, fl))
    dset = M.get_dataset(ds)
    fold, g = M.folds(dset, ds)
    tr, idd, ood = M.split_idx(dset, "train"), M.split_idx(dset, sp["id"]), M.split_idx(dset, sp["ood"])
    f = None if fl == "full" else int(fl)
    tr_f = tr if f is None else tr[fold[tr] == f]
    rec = recipe(a)
    bs = int(rec["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_tr, tf_ev = M.transform(ds, a, True), M.transform(ds, a, False)
    print("%s %s seed %d fold %s: n_train=%d groups fold0/1=%d/%d" % (
        ds, a, s, fl, len(tr_f), len(np.unique(g[tr][fold[tr] == 0])), len(np.unique(g[tr][fold[tr] == 1]))), flush=True)
    t0 = time.time()
    if not meta_p.exists():
        set_seed(s)
        train_l = M.loader(dset, tr_f, tf_tr, bs, True)
        sel_l = M.loader(dset, idd[fold[idd] == f], tf_ev, bs, False) if sp["select"] == "id_val_best" else None
        model = get_cnn_backbone(a, num_classes=sp["n_classes"], pretrained=True).to(device)
        opt = make_optimizer(model, a, rec=rec)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
        ce = nn.CrossEntropyLoss()
        hist, best_acc, best_ep = [], -1.0, 0
        for ep in range(1, NUM_EPOCHS + 1):
            te = time.time()
            model.train()
            run, n = 0.0, 0
            for x, y, _ in train_l:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = ce(model(x), y)
                loss.backward()
                opt.step()
                run += loss.item() * y.size(0)
                n += y.size(0)
            sched.step()
            acc = evaluate(model, sel_l, device)[0] if sel_l is not None else float("nan")
            hist.append({"epoch": ep, "loss": run / n, "sel_acc": acc, "seconds": time.time() - te})
            print("[Epoch %02d/%02d] loss=%.4f sel_acc=%.4f %.0fs" % (ep, NUM_EPOCHS, run / n, acc, hist[-1]["seconds"]), flush=True)
            if sel_l is not None and acc >= best_acc:
                best_acc, best_ep = acc, ep
                torch.save(model.state_dict(), best)
        torch.save(model.state_dict(), last)
        meta_p.write_text(json.dumps({"ds": ds, "arch": a, "seed": s, "fold": fl, "n_train": int(len(tr_f)),
                                      "best_epoch": best_ep, "best_sel_acc": best_acc, "recipe": rec,
                                      "history": hist}, indent=2) + "\n")
        del model
    t_train = time.time() - t0
    tm = json.loads(meta_p.read_text())
    ck = best if sp["select"] == "id_val_best" else last

    t1 = time.time()
    model = get_cnn_backbone(a, num_classes=sp["n_classes"], pretrained=False)
    model.load_state_dict(torch.load(ck, map_location=device))
    model.to(device)
    if f is None:
        extract(model, dset, ds, a, bs, device, M.feat_dir(ds) / ("%s_s42_full.npz" % a))
        print("full model + indexed features done in %.0f s" % (time.time() - t0), flush=True)
        return 0
    y = np.asarray(dset.y_array).astype(np.int64)
    tr_lo, tr_fe = extract_features_and_logits(model, M.loader(dset, tr_f, tf_ev, bs, False), device)
    id_lo, id_fe = extract_features_and_logits(model, M.loader(dset, idd, tf_ev, bs, False), device)
    oo_lo, oo_fe = extract_features_and_logits(model, M.loader(dset, ood, tf_ev, bs, False), device)
    w, b = fc_params(model)
    np.savez(M.feat_dir(ds) / ("%s_s%d_f%d.npz" % (a, s, f)), train_feats=tr_fe, train_logits=tr_lo,
             train_labels=y[tr_f], id_feats=id_fe, id_logits=id_lo, id_labels=y[idd], id_fold=fold[idd],
             id_group=g[idd], ood_feats=oo_fe, ood_logits=oo_lo, ood_labels=y[ood],
             fc_weight=np.asarray(w), fc_bias=np.asarray(b))
    t_extract = time.time() - t1
    sm, d = fold[idd] == f, fold[idd] == 1 - f
    out = {"ds": ds, "arch": a, "seed": s, "fold": f, "n_train": int(len(tr_f)), "n_id_seen": int(sm.sum()),
           "n_id_unseen": int(d.sum()), "n_ood": int(len(ood)), "best_epoch": tm["best_epoch"],
           "acc_id_seen": float((id_lo[sm].argmax(1) == y[idd][sm]).mean()),
           "acc_id_unseen": float((id_lo[d].argmax(1) == y[idd][d]).mean()),
           "acc_ood": float((oo_lo.argmax(1) == y[ood]).mean()),
           "train_seconds": round(t_train, 1), "extract_seconds": round(t_extract, 1),
           "epoch_seconds": [h["seconds"] for h in tm["history"]]}
    for name, fn in (("msp", OODScorer.score_msp), ("energy", lambda z: OODScorer.score_energy(z, 1.0))):
        so = fn(oo_lo)
        out["auroc_%s_seen" % name] = calc_auroc(fn(id_lo[sm]), so)
        out["auroc_%s_unseen" % name] = calc_auroc(fn(id_lo[d]), so)
        out["gap_%s" % name] = out["auroc_%s_seen" % name] - out["auroc_%s_unseen" % name]
    (M.report_dir(ds) / ("retrain_%s_s%d_f%d.json" % (a, s, f))).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
