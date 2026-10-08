#!/usr/bin/env python3
"""R3 item 8 / P2-a preprocessing parity audit: raw format / mode / size and 8-bit grey statistics per source (NIH sample of
1,000 = first 250 of archives 1, 4, 7, 10; all Shenzhen; all Kermany pediatric) and of the staged 256 px images.
Usage: item8_p2a_parity_audit.py results/r3/8/p2a/preproc_parity.json"""
import io, json, tarfile, zipfile, collections, sys
from pathlib import Path
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[2]
out = {}


def census(name, it):
    modes, sizes, stats, raw_stats = collections.Counter(), [], [], []
    for k, b in it:
        im = Image.open(io.BytesIO(b))
        modes[(im.format, im.mode, im.info.get("bits") or "")] += 1
        sizes.append(im.size)
        a = np.asarray(im)
        raw_stats.append((a.dtype.str, int(a.max()), a.ndim))
        g = np.asarray(im.convert("L"), dtype=np.float64)
        stats.append((g.mean(), g.std(), (g >= 255).mean(), (g <= 0).mean()))
    s = np.array(stats)
    w = np.array([x[0] for x in sizes]); h = np.array([x[1] for x in sizes])
    out[name] = {"n": len(sizes), "format_mode": {str(k): v for k, v in modes.items()},
                 "raw_dtype_max_ndim": {str(k): v for k, v in collections.Counter(raw_stats).most_common(8)},
                 "w_range": [int(w.min()), int(np.median(w)), int(w.max())],
                 "h_range": [int(h.min()), int(np.median(h)), int(h.max())],
                 "aspect_w_over_h_median": float(np.median(w / h)),
                 "L_mean": float(s[:, 0].mean()), "L_std": float(s[:, 1].mean()),
                 "frac_px_255": float(s[:, 2].mean()), "frac_px_0": float(s[:, 3].mean())}
    print(name, json.dumps(out[name]), flush=True)


def nih(n_per=250):
    for arc in sorted((R / "data/raw/nih_cxr14").glob("images_*.tar.gz"))[::3]:
        c = 0
        with tarfile.open(arc, "r|gz") as t:
            for m in t:
                if m.isfile() and m.name.endswith(".png"):
                    yield m.name, t.extractfile(m).read(); c += 1
                    if c >= n_per:
                        break


def shz():
    for f in sorted((R / "data/raw/shenzhen/CXR_png").glob("*.png")):
        yield f.name, f.read_bytes()


def kped():
    z = zipfile.ZipFile(R / "data/raw/medbench/kermany_v2/ChestXRay2017.zip")
    for n in z.namelist():
        if n.lower().endswith((".jpeg", ".jpg", ".png")) and "__MACOSX" not in n:
            yield n, z.read(n)


def staged(tarp, n=None):
    c = 0
    with tarfile.open(tarp) as t:
        for m in t:
            if m.isfile():
                yield m.name, t.extractfile(m).read(); c += 1
                if n and c >= n:
                    break


census("nih_raw_sample1000", nih())
census("shenzhen_raw", shz())
census("kermany_raw", kped())
census("nih_256_staged_sample1000", (x for a in sorted((R / "data/staged/nih_256").glob("*.tar"))[::3] for x in staged(a, 250)))
census("shenzhen_256_staged", staged(R / "data/staged/p2a_ood_256/shenzhen.tar"))
census("kermany_256_staged", staged(R / "data/staged/p2a_ood_256/kermany_ped.tar"))
json.dump(out, open(sys.argv[1], "w"), indent=1)
