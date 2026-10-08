#!/usr/bin/env python3
"""R3 item 8 / P2-d frozen foundation-model features for every image of the primary candidate (all roles of all variants).

dinov2_vitb14: local torch.hub DINOv2 ViT-B/14, x_norm_clstoken; Resize(224, bicubic) + CenterCrop(224) + ImageNet
               normalisation (pilot transform).
biomedclip:    microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (results/r3/8/p2d/g0_biomedclip.json), offline,
               model.encode_image with the model's own open_clip eval processor; run from the separate open_clip venv.
Input = staged 256 px PNGs (data/staged/p2d_*_256), RGB. Writes outputs/rigor_pack/r3_item8/p2d/{ds}_{fm}.npz (keys, feats)
and results/r3/8/p2d/train/{ds}_{fm}.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/r3")]
from item8_p2d_common import FEAT, P2D, STAGE, candidate_table  # noqa: E402

BC = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["kvasir", "brain"])
    ap.add_argument("--fm", required=True, choices=["dinov2_vitb14", "biomedclip"])
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max-n", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    import torch
    from PIL import Image

    t, cand = candidate_table(a.ds)
    if t is None:
        print("no primary candidate for", a.ds)
        return 0
    keys = sorted(set(t.key))
    if a.max_n:
        keys = sorted(np.random.default_rng(0).choice(keys, a.max_n, replace=False))
    want = set(keys)
    dev = torch.device("cuda")
    if a.fm == "dinov2_vitb14":
        from torchvision import transforms as T
        repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        model = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=True).eval().to(dev)
        tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        fwd = lambda x: model.forward_features(x)["x_norm_clstoken"]  # noqa: E731
        weights = "torch.hub facebookresearch_dinov2 dinov2_vitb14_pretrain.pth"
    else:
        import open_clip
        model, tf = open_clip.create_model_from_pretrained(BC)
        model = model.eval().to(dev)
        fwd = model.encode_image
        weights = BC + " (revision in g0_biomedclip.json)"
    t0, feats, batch, bkeys = time.time(), {}, [], []

    def flush():
        with torch.no_grad():
            f = fwd(torch.stack(batch).to(dev)).float().cpu().numpy()
        feats.update(zip(bkeys, f))
        batch.clear()
        bkeys.clear()

    for tar in sorted(STAGE[a.ds].glob("*.tar")):
        with tarfile.open(tar) as tf_:
            for m in tf_:
                k = m.name[:-4]
                if m.isfile() and k in want and k not in feats and k not in bkeys:
                    batch.append(tf(Image.open(io.BytesIO(tf_.extractfile(m).read())).convert("RGB")))
                    bkeys.append(k)
                    if len(batch) == a.bs:
                        flush()
        print(tar.name, len(feats), round(time.time() - t0), flush=True)
    if batch:
        flush()
    miss = [k for k in keys if k not in feats]
    assert not miss, ("missing", len(miss), miss[:5])
    tag = f"{a.ds}_{a.fm}" + ("_smoke" if a.max_n else "")
    FEAT.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT / f"{tag}.npz", keys=np.array(keys), feats=np.stack([feats[k] for k in keys]).astype(np.float32))
    res = {"ds": a.ds, "candidate": cand, "fm": a.fm, "weights": weights, "n": len(keys),
           "d": int(len(next(iter(feats.values())))), "input": "224 from staged 256", "torch": torch.__version__,
           "seconds": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0)}
    (P2D / "train").mkdir(parents=True, exist_ok=True)
    (P2D / "train" / f"{tag}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
