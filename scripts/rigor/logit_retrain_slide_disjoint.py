#!/usr/bin/env python3
"""Slide-disjoint retrain for the pure-logit scores (decisions/precommit_isbi_patch_2026-10-05.md, B).

One (arch, fold) per call: the seed-42 recipe of camelyon17_pilot.py, unchanged, trained on the train
patches of the fold's slides only (H11c folds); checkpoint selected on the id_val patches of the
same fold's slides. Then MSP / Energy (T=1) AUROC with OOD = all hospital 2 and
  ID = id_val patches of the other fold's slides   (slide-disjoint, primary)
  ID = id_val patches of the same fold's slides    (same-slide contrast, secondary)

Checkpoints: data/models/camelyon17/frac1/slide_disjoint/seed42/{arch}_fold{f}_best.pth
Logits:      outputs/rigor_pack/logit_retrain_slide_disjoint/{arch}_fold{f}_logits.npz
Result:      outputs/reports/rigor_pack/leakage/logit_retrain_slide_disjoint/{arch}_fold{f}.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from leakfree_fit_scores import REPRO_FOLD_SIZES, folds  # noqa: E402

SEED = 42


def load_pilot():
    spec = importlib.util.spec_from_file_location("camelyon17_pilot", C.REPO / "scripts" / "camelyon17_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["resnet50", "convnext_tiny", "densenet121"])
    ap.add_argument("--fold", type=int, required=True, choices=[0, 1])
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from wilds.datasets.wilds_dataset import WILDSSubset

    from src.datasets.camelyon_ood import _WildsPathDataset, build_transform, get_wilds_dataset
    from src.utils.benchmark_metrics import calc_auroc
    from src.utils.scoring import OODScorer

    P = load_pilot()
    a, f = args.arch, args.fold
    mdir = C.ensure_dir(C.REPO / ("data/models/camelyon17/frac1/slide_disjoint/seed%d" % SEED))
    ldir = C.ensure_dir(C.default_out(C.REPO) / "logit_retrain_slide_disjoint")
    rdir = C.ensure_dir(C.default_reports(C.REPO) / "leakage/logit_retrain_slide_disjoint")
    best, meta_p = mdir / ("%s_fold%d_best.pth" % (a, f)), mdir / ("%s_fold%d_train_meta.json" % (a, f))

    meta = C.load_camelyon_metadata(C.REPO)
    tr_idx = np.where(meta.wilds_split.to_numpy() == 0)[0]
    iv_idx = np.where(meta.wilds_split.to_numpy() == 1)[0]
    tf, vf = folds(meta.slide.to_numpy()[tr_idx], meta.center.to_numpy()[tr_idx], meta.slide.to_numpy()[iv_idx], SEED)
    sizes = (int((tf == 0).sum()), int((tf == 1).sum()))
    if sizes != REPRO_FOLD_SIZES:
        raise SystemExit("STOP: fold sizes %s != H11c %s" % (sizes, REPRO_FOLD_SIZES))

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
    epoch_s = []
    if not meta_p.exists():
        P.set_seed(SEED)
        train_l = mk(tr_idx[tf == f], train_tf, "camelyon17/train", True)
        sel_l = mk(iv_idx[vf == f], eval_tf, "camelyon17/id_val", False)
        model = P.get_model(a).to(device)
        crit = nn.CrossEntropyLoss()
        opt = P.make_optimizer(model, a, rec=rec)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.NUM_EPOCHS)
        n_train = len(train_l.dataset)
        print("=== slide-disjoint retrain %s fold %d seed %d n_train=%d n_sel=%d optim=%s lr=%s wd=%s bs=%d ===" % (
            a, f, SEED, n_train, len(sel_l.dataset), rec["optim"], rec["lr"], rec["wd"], bs), flush=True)
        best_acc = -1.0
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
            epoch_s.append(time.time() - te)
            print("[Epoch %02d/%02d] loss=%.4f sel_acc=%.4f %.0fs" % (ep, P.NUM_EPOCHS, run / n_train, acc, epoch_s[-1]), flush=True)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best)
        meta_p.write_text(json.dumps({"arch": a, "fold": f, "seed": SEED, "n_train": n_train,
                                      "best_sel_acc": best_acc, "optim": rec["optim"], "lr": rec["lr"],
                                      "wd": rec["wd"], "batch_size": bs, "epoch_seconds": epoch_s}, indent=2) + "\n")
        del model
    t_train = time.time() - t0

    model = P.get_model(a, pretrained=False)
    model.load_state_dict(torch.load(best, map_location=device))
    model.to(device)
    iv_logits = P.collect_logits(model, mk(iv_idx, eval_tf, "camelyon17/id_val", False), device)
    oo = ds.get_subset("test", transform=eval_tf)
    oo_logits = P.collect_logits(model, DataLoader(_WildsPathDataset(oo, "camelyon17/test"), batch_size=bs, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=True), device)
    np.savez(ldir / ("%s_fold%d_logits.npz" % (a, f)), id_val=iv_logits, ood=oo_logits, id_val_fold=vf)
    out = {"arch": a, "fold": f, "seed": SEED, "n_train": sizes[f],
           "n_id_disjoint": int((vf == 1 - f).sum()), "n_id_same": int((vf == f).sum()), "n_ood": len(oo_logits),
           "id_val_acc_disjoint": float((iv_logits[vf == 1 - f].argmax(1) == meta.tumor.to_numpy()[iv_idx][vf == 1 - f]).mean()),
           "train_seconds": round(t_train, 1), "epoch_seconds": epoch_s or json.loads(meta_p.read_text())["epoch_seconds"]}
    for name, fn in (("msp", OODScorer.score_msp), ("energy", lambda z: OODScorer.score_energy(z, 1.0))):
        so = fn(oo_logits)
        out["auroc_%s_slide_disjoint" % name] = calc_auroc(fn(iv_logits[vf == 1 - f]), so)
        out["auroc_%s_same_slides" % name] = calc_auroc(fn(iv_logits[vf == f]), so)
    out["seconds_total"] = round(time.time() - t0, 1)
    (rdir / ("%s_fold%d.json" % (a, f))).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
