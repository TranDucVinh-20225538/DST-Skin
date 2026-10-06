#!/usr/bin/env python3
"""M1 of decisions/precommit_leak_mechanism_fix_2026-10-06.md (GPU): re-extract features / logits of
existing models with the input appearance ablated, no retraining.

Variants (applied to the native PIL image before the usual transform):
  macenko : Camelyon only, Macenko et al. 2009 (Io 240, alpha 1, beta 0.15, standard HERef / maxCRef);
            tissue pixel = OD > beta in all channels; < 10% tissue or failed eigen-decomposition -> unchanged, counted
  colour  : iWildCam / RxRx1, per-image per-channel z-score mapped to ImageNet mean / std
  gray    : ITU-R 601 luma replicated to 3 channels
--model published : seed-42 standard model; train (as the published features: Camelyon train under the
                    train augmentation, multibench under the eval transform) + ID + OOD, same order as
                    the published / indexed features.
--model retrained : seed-42 group-disjoint retrained model of fold --fold; ID + OOD only.
Output: outputs/rigor_pack/mechanism_fix/m1/{ds}_{model}_{arch}[_f{fold}]_{variant}.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

REPO = C.REPO
OUT = REPO / "outputs/rigor_pack/mechanism_fix/m1"
HEREF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAXCREF = np.array([1.9705, 1.0308])
IMNET_MEAN, IMNET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def macenko(img: np.ndarray, Io=240.0, alpha=1.0, beta=0.15):
    h, w, _ = img.shape
    I = img.reshape(-1, 3).astype(np.float64)
    OD = -np.log((I + 1.0) / Io)
    tissue = np.all(OD > beta, axis=1)
    if tissue.mean() < 0.10:
        return img, 1
    try:
        ODhat = OD[tissue]
        _, V = np.linalg.eigh(np.cov(ODhat.T))
        V = V[:, 1:3]
        That = ODhat @ V
        phi = np.arctan2(That[:, 1], That[:, 0])
        lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
        vmin = V @ np.array([np.cos(lo), np.sin(lo)])
        vmax = V @ np.array([np.cos(hi), np.sin(hi)])
        HE = np.array([vmin, vmax]).T if vmin[0] > vmax[0] else np.array([vmax, vmin]).T
        Cc = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        maxC = np.array([np.percentile(Cc[0], 99), np.percentile(Cc[1], 99)])
        Cc = Cc / maxC[:, None] * MAXCREF[:, None]
        out = Io * np.exp(-HEREF @ Cc)
        if not np.all(np.isfinite(out)):
            return img, 1
        return np.clip(out.T, 0, 255).reshape(h, w, 3).astype(np.uint8), 0
    except np.linalg.LinAlgError:
        return img, 1


def colour_std(img: np.ndarray):
    x = img.astype(np.float64) / 255.0
    m, s = x.reshape(-1, 3).mean(0), x.reshape(-1, 3).std(0)
    z = (x - m) / np.maximum(s, 1e-6)
    return (np.clip(z * IMNET_STD + IMNET_MEAN, 0, 1) * 255).astype(np.uint8), 0


def apply(variant: str, im):
    from PIL import Image
    im = im.convert("RGB")
    if variant == "gray":
        return im.convert("L").convert("RGB"), 0
    a, fail = (macenko if variant == "macenko" else colour_std)(np.asarray(im))
    return Image.fromarray(a), fail


class Ablated:
    def __init__(self, subset, variant, tfm):
        self.s, self.v, self.t = subset, variant, tfm

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        x, y, _ = self.s[i]
        im, fail = apply(self.v, x)
        return self.t(im), int(y), fail


def extract(model, loader, device):
    import torch
    layer = model.avgpool if hasattr(model, "avgpool") else model.features[-1]
    buf = []
    h = layer.register_forward_hook(lambda m, i, o: buf.append(o.flatten(1).detach()))
    L, Fe, fails = [], [], 0
    model.eval()
    with torch.no_grad():
        for x, _, f in loader:
            lo = model(x.to(device))
            L.append(lo.cpu().numpy())
            Fe.append(buf.pop().cpu().numpy())
            fails += int(f.sum())
    h.remove()
    return np.concatenate(L).astype(np.float32), np.concatenate(Fe).astype(np.float32), fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["camelyon", "iwildcam", "rxrx1"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--variant", required=True, choices=["macenko", "colour", "gray"])
    ap.add_argument("--model", required=True, choices=["published", "retrained"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from wilds.datasets.wilds_dataset import WILDSSubset

    from src.models.cnn_family import fc_params, recipe
    ds, a, v = args.ds, args.arch, args.variant
    if (v == "macenko" and ds != "camelyon") or (v == "colour" and ds == "camelyon"):
        raise SystemExit("macenko is for Camelyon, colour for iWildCam / RxRx1")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{ds}_{args.model}_{a}" + (f"_f{args.fold}" if args.model == "retrained" else "") + f"_{v}"
    dst = OUT / f"{name}.npz"
    if dst.exists():
        print("exists", dst)
        return 0
    dev = torch.device("cuda")
    bs = int(recipe(a)["batch_size"])
    t0 = time.time()
    splits = {}
    if ds == "camelyon":
        from extract_features_indexed import ckpt_path, input_size_for, load_pilot
        from src.datasets.camelyon_ood import build_transform, get_wilds_dataset
        P = load_pilot()
        dset = get_wilds_dataset(download=False)
        ref = torch.load(C.feature_path(REPO, "camelyon17", a if args.model == "published" else "resnet18", 42, indexed=True),
                         map_location="cpu", weights_only=False)
        isz = input_size_for(a) if args.model == "published" else None
        ev = build_transform(a, train=False, input_size=isz)
        if args.model == "published":
            splits["train"] = (np.asarray(ref["train_idx"]), build_transform(a, train=True, input_size=isz))
            ck = ckpt_path(REPO, a, 42)
        else:
            ck = REPO / f"data/models/camelyon17/frac1/slide_disjoint_v2/seed42/{a}_fold{args.fold}_best.pth"
        splits["id"] = (np.asarray(ref["val_idx"]), ev)
        splits["ood"] = (np.asarray(ref["ood_idx"]), ev)
        model = P.get_model(a, pretrained=False)
    else:
        import multibench_common as M
        from src.models.cnn_family import get_cnn_backbone
        dset = M.get_dataset(ds)
        sp = M.SPECS[ds]
        ev = M.transform(ds, a, False)
        full = np.load(M.feat_dir(ds) / f"{a}_s42_full.npz")
        if args.model == "published":
            splits["train"] = (full["train_idx"], ev)
            ck = M.published_ckpt(ds, a)
        else:
            sel = "best" if sp["select"] == "id_val_best" else "last"
            ck = REPO / f"data/models/multibench/{ds}/seed42/{a}_{args.fold}_{sel}.pth"
        splits["id"] = (full["id_idx"], ev)
        splits["ood"] = (full["ood_idx"], ev)
        model = get_cnn_backbone(a, num_classes=sp["n_classes"], pretrained=False)
    model.load_state_dict(torch.load(ck, map_location=dev))
    model.to(dev)
    out = {}
    for k, (idx, tfm) in splits.items():
        loader = DataLoader(Ablated(WILDSSubset(dset, idx, None), v, tfm), batch_size=bs, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        lo, fe, fails = extract(model, loader, dev)
        out.update({f"{k}_logits": lo, f"{k}_feats": fe, f"{k}_idx": idx, f"{k}_fails": fails,
                    f"{k}_labels": np.asarray(dset.y_array[idx]).astype(np.int64)})
        print(f"{name} {k}: n={len(idx)} fails={fails} {time.time() - t0:.0f}s", flush=True)
    w, b = fc_params(model)
    out.update(fc_weight=np.asarray(w), fc_bias=np.asarray(b))
    np.savez(dst, **out)
    print("saved", dst.name, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
