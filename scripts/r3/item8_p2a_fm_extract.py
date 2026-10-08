#!/usr/bin/env python3
"""R3 item 8 / P2-a frozen foundation-model features (results/r3/8/PRECOMMIT.json): dinov2_vitb14 or rad_dino.

dinov2_vitb14: local torch.hub DINOv2 ViT-B/14, x_norm_clstoken; input = NIH / OOD staged 256 x 256 PNGs
               (data/staged/nih_256, data/staged/p2a_ood_256), Resize(224, bicubic) + CenterCrop(224), ImageNet
               normalisation (the item-8 ceiling-pilot transform).
rad_dino:      microsoft/rad-dino (Hugging Face, MIT, not gated; cached snapshot, offline), pooler_output (CLS after
               the final layer norm, as on the model card); input = original images (NIH 1024 x 1024 archives,
               Shenzhen PNGs, Kermany pediatric zip) -> grayscale, 518 x 518 bicubic (the staging squash at the
               model's 518 px), mean 0.5307 / std 0.2583 (the model's preprocessor_config.json).
Images: roles fit / seen / unseen of results/r3/8/p2a/sets.csv.gz and every OOD image of ood.csv.gz. Archives are
streamed (one worker per archive), nothing is written to node-local disk.
Writes outputs/rigor_pack/r3_item8/p2a/{fm}.npz, results/r3/8/p2a/train/{fm}.json
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P2A = REPO / "results/r3/8/p2a"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
            "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
ROLES = ("fit", "seen", "unseen")


def transform(fm):
    from torchvision import transforms as T
    if fm == "dinov2_vitb14":
        return T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return T.Compose([T.ToTensor(), T.Normalize((0.5307,) * 3, (0.2583,) * 3)])


def prep(fm, f):
    from PIL import Image
    im = Image.open(f).convert("L")
    if fm == "rad_dino":
        im = im.resize((518, 518), Image.BICUBIC)
    return im.convert("RGB")


def stream_ds(fm, archives, wanted):
    import torch

    class TarStream(torch.utils.data.IterableDataset):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            arcs = archives[wi.id::wi.num_workers] if wi else archives
            tf = transform(fm)
            for arc in arcs:
                with tarfile.open(arc, "r|gz" if arc.name.endswith(".gz") else "r|") as t:
                    for m in t:
                        name = Path(m.name).name
                        if not m.isfile():
                            continue
                        key = name[:-4] if name.endswith(".png") and name.startswith(("CHNCXR", "kped_")) else name
                        if key in wanted:
                            yield tf(prep(fm, io.BytesIO(t.extractfile(m).read()))), key
    return TarStream()


def raw_ood_ds(fm, O):
    import torch
    shz = REPO / "data/raw/shenzhen/CXR_png"
    zp = REPO / "data/raw/medbench/kermany_v2/ChestXRay2017.zip"

    class RawOOD(torch.utils.data.Dataset):
        def __len__(self):
            return len(O)

        def __getitem__(self, i):
            r = O.iloc[i]
            if r.source == "shenzhen":
                b = (shz / (r.key + ".png")).read_bytes()
            else:
                if not hasattr(self, "z"):
                    self.z = zipfile.ZipFile(zp)
                b = self.z.read(r.member)
            return transform(fm)(prep(fm, io.BytesIO(b))), r.key
    return RawOOD()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True, choices=["dinov2_vitb14", "rad_dino"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-n", type=int, default=None, help="smoke only: cap images per set")
    args = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader

    fm = args.fm
    tag = fm + ("_smoke" if args.max_n else "")
    S = pd.read_csv(P2A / "sets.csv.gz")
    O = pd.read_csv(P2A / "ood.csv.gz", keep_default_na=False)
    if args.max_n:
        S = S[S.patient < 1300]
        S = pd.concat([S[S.role == r].iloc[: args.max_n] for r in ROLES])
        O = pd.concat([O[O.source == s].iloc[: args.max_n] for s in ("shenzhen", "kermany_ped")])
    S = S[S.role.isin(ROLES)]
    dev = torch.device("cuda")
    if fm == "dinov2_vitb14":
        repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=True).eval().to(dev)
        fwd = lambda x: model.forward_features(x)["x_norm_clstoken"]  # noqa: E731
        nih = sorted((REPO / "data/staged/nih_256").glob("images_*.tar"))
        ood_loader = DataLoader(stream_ds(fm, sorted((REPO / "data/staged/p2a_ood_256").glob("*.tar")), set(O.key)),
                                batch_size=args.bs, num_workers=2)
        weights = "torch.hub facebookresearch_dinov2 dinov2_vitb14_pretrain.pth"
    else:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel
        snap = snapshot_download("microsoft/rad-dino", allow_patterns=["*.json", "*.safetensors"], local_files_only=True)
        model = AutoModel.from_pretrained(snap).eval().to(dev)
        fwd = lambda x: model(pixel_values=x).pooler_output  # noqa: E731
        nih = sorted((REPO / "data/raw/nih_cxr14").glob("images_*.tar.gz"))
        ood_loader = DataLoader(raw_ood_ds(fm, O), batch_size=args.bs, num_workers=min(args.workers, 8))
        weights = "microsoft/rad-dino snapshot " + Path(snap).name
    if args.max_n:
        nih = nih[:1]
    t0 = time.time()
    feats = {}

    def run(loader, n_expect):
        with torch.no_grad():
            for x, keys in loader:
                f = fwd(x.to(dev, non_blocking=True)).float().cpu().numpy()
                feats.update(zip(keys, f))
                if len(feats) % 5000 < len(keys):
                    print(len(feats), "/", n_expect, round(time.time() - t0), flush=True)

    n_all = len(S) + len(O)
    run(DataLoader(stream_ds(fm, nih, set(S.image)), batch_size=args.bs, num_workers=min(args.workers, len(nih)),
                   pin_memory=True), n_all)
    run(ood_loader, n_all)
    miss = [k for k in list(S.image) + list(O.key) if k not in feats]
    assert not miss, ("missing", len(miss), miss[:5])
    out = {}
    for r in ROLES:
        x = S[S.role == r]
        out[f"{r}_feats"] = np.stack([feats[k] for k in x.image]).astype(np.float32)
        out[f"{r}_labels"] = x[FINDINGS].to_numpy().astype(np.int8)
        out[f"{r}_keys"] = x.image.to_numpy()
    out["ood_feats"] = np.stack([feats[k] for k in O.key]).astype(np.float32)
    out["ood_keys"] = O.key.to_numpy()
    fdir = REPO / "outputs/rigor_pack/r3_item8/p2a"
    fdir.mkdir(parents=True, exist_ok=True)
    np.savez(fdir / f"{tag}.npz", **out)
    res = {"fm": fm, "weights": weights, "n": {k[:-6]: len(v) for k, v in out.items() if k.endswith("_feats")},
           "d": int(out["fit_feats"].shape[1]), "input": "224 from staged 256" if fm == "dinov2_vitb14" else
           "518 from originals", "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0)}
    if fm == "rad_dino":
        import transformers
        res["transformers"] = transformers.__version__
    (P2A / "train").mkdir(parents=True, exist_ok=True)
    (P2A / "train" / f"{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
