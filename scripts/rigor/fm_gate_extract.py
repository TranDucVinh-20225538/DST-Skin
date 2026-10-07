#!/usr/bin/env python3
"""Frozen foundation-model embeddings for decisions/precommit_foundation_leakage_gate_2026-10-07.md (GPU).

--fm dinov2_vitb14 | dinov2_vitl14 : torch.hub facebookresearch/dinov2 (cached repo + weights), x_norm_clstoken
--fm uni                           : src/models/pathology_fm.py spec "uni" (timm hf-hub:MahmoodLab/uni), CLS
--fm virchow2                      : timm hf-hub:paige-ai/Virchow2 (SwiGLUPacked, SiLU), concat(CLS, mean patch)
--fm conch_v1_5                    : MahmoodLab/TITAN conch_v1_5.py build_conch, its own eval transform (448)
CONCH v1 (MahmoodLab/CONCH) is gated and not accessible with the configured token (403); see access_status.json.

Every FM: frozen, eval(), float32, no autocast, no augmentation; the same eval transform for every split.
Load result (ok / STOP + reason, feat dim, transform) -> outputs/rigor_pack/foundation_gate/load/{fm}.json.
--ds camelyon   : train / id_val / hospital-2 OOD in the index order of the rigor-pack indexed features.
--ds dermamnist | isic2019 | breakhis : every image of the non-val sets of the locked medbench arms (MED_ARMS;
                  ID and OOD), read by key from the node-local copy of data/staged/{ds}_256.tar ($DST_IMG_DIR).
--limit n       : smoke (first n images of every split), throughput only.
Output: outputs/rigor_pack/foundation_gate/feats/{ds}_{fm}[_smoke].npz
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

REPO = C.REPO
OUT = REPO / "outputs/rigor_pack/foundation_gate"
FMS = ("dinov2_vitb14", "uni", "conch_v1_5", "virchow2", "dinov2_vitl14")
BATCH = {"dinov2_vitb14": 256, "dinov2_vitl14": 128, "uni": 128, "conch_v1_5": 64, "virchow2": 64}


def tf224():
    from src.models.pathology_fm import SPECS, eval_transform
    return eval_transform(SPECS["uni"])


def load_fm(name: str):
    """Returns (forward fn, transform, info)."""
    import torch
    if name.startswith("dinov2"):
        repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        m = torch.hub.load(str(repo), name, source="local", pretrained=True)
        fwd = lambda x: m.forward_features(x)["x_norm_clstoken"]  # noqa: E731
        info = {"repo": "facebookresearch/dinov2 (torch.hub %s)" % name, "embedding": "x_norm_clstoken",
                "transform": "Resize 224 bicubic + CenterCrop 224 + ImageNet mean/std"}
        tf = tf224()
    elif name == "uni":
        from src.models.pathology_fm import load_frozen_encoder
        m, spec = load_frozen_encoder("uni")
        fwd = m
        info = {"repo": spec.timm_id, "embedding": "CLS (timm num_classes=0)",
                "transform": "Resize 224 bicubic + CenterCrop 224 + ImageNet mean/std"}
        tf = tf224()
    elif name == "virchow2":
        import timm
        from timm.layers import SwiGLUPacked
        m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked,
                              act_layer=torch.nn.SiLU)

        def fwd(x):
            o = m(x)
            return torch.cat([o[:, 0], o[:, 5:].mean(1)], dim=1)
        info = {"repo": "hf-hub:paige-ai/Virchow2", "embedding": "concat(CLS, mean patch tokens after 4 registers)",
                "transform": "Resize 224 bicubic + CenterCrop 224 + ImageNet mean/std"}
        tf = tf224()
    elif name == "conch_v1_5":
        from types import SimpleNamespace

        from huggingface_hub import hf_hub_download
        src = hf_hub_download("MahmoodLab/TITAN", "conch_v1_5.py")
        cfg = json.loads(Path(hf_hub_download("MahmoodLab/TITAN", "config.json")).read_text())["conch_config"]
        spec = importlib.util.spec_from_file_location("conch_v1_5", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m, tf = mod.build_conch(SimpleNamespace(**cfg))
        fwd = m
        info = {"repo": "MahmoodLab/TITAN conch_v1_5_pytorch_model.bin (CONCH v1.5)",
                "embedding": "attentional-pool contrast embedding (EncoderWithAttentionalPooler.forward)",
                "transform": "Resize 448 bilinear + CenterCrop 448 + ImageNet mean/std (build_conch)"}
    else:
        raise ValueError(name)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    info["n_params"] = int(sum(p.numel() for p in m.parameters()))
    return m, fwd, tf, info


class Keyed:
    """medbench images from the node-local copy of the staged tar, by key."""

    def __init__(self, img_dir, keys, tfm):
        import medbench_common as M
        self.p = [Path(img_dir) / M.member(k) for k in keys]
        self.t = tfm

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.p[i]) as im:
            return self.t(im.convert("RGB")), -1


def run(fwd, loader, device):
    import torch
    out, ys = [], []
    with torch.no_grad():
        for b in loader:
            x, y = b[0], b[1]
            out.append(fwd(x.to(device, non_blocking=True)).float().cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(out).astype(np.float32), np.concatenate(ys).astype(np.int64)


def camelyon_splits(tfm, limit):
    import torch
    from wilds.datasets.wilds_dataset import WILDSSubset

    from src.datasets.camelyon_ood import get_wilds_dataset
    dset = get_wilds_dataset(download=False)
    ref = torch.load(C.feature_path(REPO, "camelyon17", "resnet18", 42, indexed=True), map_location="cpu",
                     weights_only=False)
    out = {}
    for k, rk in (("train", "train"), ("id", "val"), ("ood", "ood")):
        n = len(ref["%s_idx" % rk])
        sel = np.arange(n) if not limit else np.arange(0, n, max(1, n // limit))[:limit]
        idx = np.asarray(ref["%s_idx" % rk])[sel]
        out[k] = (WILDSSubset(dset, idx, tfm), {"%s_idx" % k: idx, "%s_slide" % k: np.asarray(ref["%s_slide" % rk])[sel],
                                                 "ref_labels": np.asarray(ref["%s_labels" % rk])[sel]})
    return out


MED_ARMS = {"dermamnist": ("std", "b0", "b1"), "isic2019": ("std", "b0", "b1"),
            "breakhis": tuple("std_r%d" % r for r in range(5)) + tuple("gd_r%d" % r for r in range(5))}


def medical_splits(ds, tfm, limit):
    """One split "all": the union of the keys of every non-val set of the dataset's locked medbench arms."""
    import os

    import medbench_common as M
    keys = sorted({k for a in MED_ARMS[ds] for s, v in M.arm(ds, a).items() if s != "val" for k in v["keys"]})
    keys = keys[::max(1, len(keys) // limit)][:limit] if limit else keys
    return {"all": (Keyed(os.environ["DST_IMG_DIR"], keys, tfm), {"all_keys": np.asarray(keys)})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True, choices=FMS)
    ap.add_argument("--ds", required=True, choices=["camelyon", "dermamnist", "isic2019", "breakhis"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(8)
    (OUT / "load").mkdir(parents=True, exist_ok=True)
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    name = "%s_%s%s" % (args.ds, args.fm, "_smoke" if args.limit else "")
    dst = OUT / "feats" / ("%s.npz" % name)
    if dst.exists():
        print("exists", dst.name)
        return 0
    dev = torch.device("cuda")
    t0 = time.time()
    rec = {"fm": args.fm, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    try:
        m, fwd, tfm, info = load_fm(args.fm)
        m.to(dev)
        with torch.no_grad():
            d = fwd(torch.zeros(2, 3, 224 if args.fm != "conch_v1_5" else 448, 224 if args.fm != "conch_v1_5" else 448,
                                device=dev))
        rec.update(info, status="ok", feat_dim=int(d.shape[1]), load_seconds=round(time.time() - t0, 1))
    except Exception as e:  # load failure is a STOP reason for this FM, not a crash of the grid
        rec.update(status="STOP", reason="%s: %s" % (type(e).__name__, str(e)[:500]))
        (OUT / "load" / ("%s.json" % args.fm)).write_text(json.dumps(rec, indent=2) + "\n")
        print("STOP", rec["reason"], flush=True)
        return 0
    (OUT / "load" / ("%s.json" % args.fm)).write_text(json.dumps(rec, indent=2) + "\n")
    print("loaded", json.dumps(rec), flush=True)
    splits = camelyon_splits(tfm, args.limit) if args.ds == "camelyon" else medical_splits(args.ds, tfm, args.limit)
    out, n_img = {}, 0
    for k, (sub, extra) in splits.items():
        t1 = time.time()
        loader = DataLoader(sub, batch_size=BATCH[args.fm], shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=False)
        fe, y = run(fwd, loader, dev)
        if "ref_labels" in extra and not np.array_equal(extra.pop("ref_labels"), y):
            raise SystemExit("STOP: %s labels differ from the indexed reference order" % k)
        out.update({"%s_feats" % k: fe, "%s_labels" % k: y}, **extra)
        n_img += len(y)
        print("%s %s: n=%d dim=%d %.0fs (%.0f img/s)" % (name, k, len(y), fe.shape[1], time.time() - t1,
                                                          len(y) / max(time.time() - t1, 1e-6)), flush=True)
    np.savez(dst, **out)
    el = time.time() - t0
    print("saved %s n=%d %.0fs overall %.1f img/s" % (dst.name, n_img, el, n_img / el), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
