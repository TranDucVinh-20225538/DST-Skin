#!/usr/bin/env python3
"""Medbench staging: decode every image a dataset needs once, resize the short side to 256 (224-px
archs: the eval Resize of the recipe) and 343 (EfficientNet-B3 @300: round(300*256/224)), JPEG q95,
write data/staged/<ds>_256.tar and data/staged/<ds>_343.tar. Member name = sha1(key)[:20].jpg,
key as in outputs/reports/rigor_pack/medbench/splits/*.csv.gz (prefixed by source, see KEYS below).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw/medbench"
SPL = REPO / "outputs/reports/rigor_pack/medbench/splits"
SIZES = (256, 343)


def member(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:20] + ".jpg"


def to_rgb(im: Image.Image) -> Image.Image:
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im, dtype=np.float64)
        lo, hi = a.min(), a.max()
        im = Image.fromarray(((a - lo) / max(hi - lo, 1e-8) * 255).astype(np.uint8))
    return im.convert("RGB")


def resize_short(im: Image.Image, s: int) -> Image.Image:
    w, h = im.size
    r = s / min(w, h)
    return im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR)


def work(item):
    key, payload = item
    im = payload if isinstance(payload, Image.Image) else Image.open(io.BytesIO(payload))
    im = to_rgb(im)
    out = {}
    for s in SIZES:
        b = io.BytesIO()
        resize_short(im, s).save(b, format="JPEG", quality=95)
        out[s] = b.getvalue()
    return key, out


def npz_items(path, split, prefix):
    z = np.load(path)
    for i, a in enumerate(z[f"{split}_images"]):
        yield f"{prefix}{split}:{i}", Image.fromarray(a)


def zip_items(path, names, prefix=""):
    zf = zipfile.ZipFile(path)
    for n in names:
        yield prefix + n, zf.read(n)


def tar_items(path, keep, prefix=""):
    with tarfile.open(path, "r|gz") as t:
        for m in t:
            if m.isfile() and m.name in keep:
                yield prefix + m.name, t.extractfile(m).read()


def sources(ds: str):
    if ds == "dermamnist":
        for s in ("train", "val", "test"):
            yield from npz_items(RAW / "medmnist/dermamnist_224.npz", s, "")
        yield from npz_items(RAW / "dermamnist_ce/dermamnist_extended_224.npz", "test", "E:")
        for s in ("train", "val", "test"):
            yield from npz_items(RAW / "dermamnist_ce/dermamnist_corrected_224.npz", s, "C:")
        yield from npz_items(RAW / "medmnist/bloodmnist_224.npz", "test", "blood:")
        yield from pad_items()
        isic = pd.read_csv(SPL / "isic2019.csv.gz")
        bcn = isic.image[isic.source == "BCN"]
        yield from isic_items(set(bcn))
    elif ds == "isic2019":
        isic = pd.read_csv(SPL / "isic2019.csv.gz")
        yield from isic_items(set(isic.image))
        yield from pad_items()
    elif ds == "kermany":
        k = pd.read_csv(SPL / "kermany.csv.gz")
        v2 = set(k.key[k.version == "v2"])
        yield from tar_items(RAW / "kermany_v2/OCT2017.tar.gz", v2, "v2:")
        zf = RAW / "kermany_v3/ZhangLabData.zip"
        yield from zip_items(zf, list(k.key[k.version == "v3"]), "v3:")
    elif ds == "breakhis":
        b = pd.read_csv(SPL / "breakhis.csv.gz")
        yield from tar_items(RAW / "breakhis/BreaKHis_v1.tar.gz", set(b.key), "")
        c = pd.read_csv(SPL / "crc_val.csv.gz")
        yield from zip_items(RAW / "crc_val/CRC-VAL-HE-7K.zip", list(c.key), "")
    elif ds == "brain_cheng":
        import h5py
        for z in sorted((RAW / "brain_cheng").glob("figshare_*.zip")):
            zf = zipfile.ZipFile(z)
            for n in zf.namelist():
                if n.endswith(".mat"):
                    with h5py.File(io.BytesIO(zf.read(n)), "r") as h:
                        a = np.asarray(h["cjdata"]["image"]).T.astype(np.float64)
                    a = ((a - a.min()) / max(a.max() - a.min(), 1e-8) * 255).astype(np.uint8)
                    yield f"{z.name}:{n}", Image.fromarray(a)
        k = pd.read_csv(SPL / "kermany.csv.gz")
        yield from zip_items(RAW / "kermany_v3/ZhangLabData.zip", list(k.key[k.label == "CXR"]), "v3:")


def pad_items():
    d = REPO / "data/raw/pad_ufes20/images"
    for p in sorted(d.rglob("*.png")):
        yield "pad:" + p.name, p.read_bytes()


def isic_items(keep: set):
    zf = zipfile.ZipFile(RAW / "isic2019/ISIC_2019_Training_Input.zip")
    for n in zf.namelist():
        st = Path(n).stem
        if n.endswith(".jpg") and st in keep:
            yield "isic:" + st, zf.read(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = REPO / "data/staged"
    out.mkdir(parents=True, exist_ok=True)
    tmp = {s: out / f"{args.ds}_{s}.tar.part" for s in SIZES}
    tars = {s: tarfile.open(tmp[s], "w") for s in SIZES}
    keys = []
    with Pool(args.workers) as pool:
        for n, (key, enc) in enumerate(pool.imap(work, sources(args.ds), chunksize=16)):
            for s in SIZES:
                ti = tarfile.TarInfo(member(key))
                ti.size = len(enc[s])
                tars[s].addfile(ti, io.BytesIO(enc[s]))
            keys.append(key)
            if n % 5000 == 0:
                print(n, key, flush=True)
    for s in SIZES:
        tars[s].close()
        tmp[s].rename(out / f"{args.ds}_{s}.tar")
    pd.DataFrame({"key": keys, "member": [member(k) for k in keys]}).to_csv(out / f"{args.ds}_manifest.csv.gz", index=False)
    print("staged", args.ds, len(keys), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
