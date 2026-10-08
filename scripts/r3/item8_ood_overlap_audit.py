#!/usr/bin/env python3
"""R3 item 8 (post hoc): OOD / ID overlap audit for Kermany OCT and ISIC 2019, medbench (B) arm b0 / b1 (P2-b) and
Track A arm std (R3 item 1).

  hash  <ds>  : sha1 of the decoded pixels (mode, shape, bytes) of every image of the split table, read from the raw
               archives (kermany: v2 OCT2017.tar.gz and v3 ZhangLabData.zip; isic2019: ISIC_2019_Training_Input.zip).
               Writes ~/r3work/item8_oodaudit/hashes_<ds>.csv.gz (not committed: derived from raw data).
  audit       : per arm, OOD groups / images that share a group with each ID set (train, val, seen, unseen and, for
               Kermany std, unseen_extra), the OOD images whose group is absent from every ID set ("clean OOD"),
               and pixel-identical OOD / ID-set pairs. Group = patient ID (Kermany) or lesion_id / image (ISIC 2019,
               which has no patient ID). Writes results/r3/8/ood_overlap/audit.json.

Arm definitions follow scripts/rigor/medbench_common.arm.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SPL = REPO / "outputs/reports/rigor_pack/medbench/splits"
RAW = REPO / "data/raw/medbench"
WORK = Path.home() / "r3work/item8_oodaudit"
OUT = REPO / "results/r3/8/ood_overlap"


def pixhash(item):
    key, b = item
    im = Image.open(io.BytesIO(b))
    a = np.asarray(im)
    return key, hashlib.sha1(f"{im.mode}|{a.shape}|".encode() + a.tobytes()).hexdigest(), im.mode, "x".join(map(str, a.shape))


def stream(ds):
    if ds == "isic2019":
        z = zipfile.ZipFile(RAW / "isic2019/ISIC_2019_Training_Input.zip")
        for n in z.namelist():
            if n.endswith(".jpg"):
                yield "isic:" + Path(n).stem, z.read(n)
        return
    keys = set(pd.read_csv(SPL / "kermany.csv.gz", low_memory=False, usecols=["version", "key"]).pipe(
        lambda d: d.version + ":" + d.key))
    with tarfile.open(RAW / "kermany_v2/OCT2017.tar.gz", "r|gz") as t:
        for m in t:
            if m.isfile() and "v2:" + m.name in keys:
                yield "v2:" + m.name, t.extractfile(m).read()
    z = zipfile.ZipFile(RAW / "kermany_v3/ZhangLabData.zip")
    for n in z.namelist():
        if "v3:" + n in keys:
            yield "v3:" + n, z.read(n)


def do_hash(ds):
    WORK.mkdir(parents=True, exist_ok=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as p:
        rows = list(p.imap(pixhash, stream(ds), chunksize=64))
    h = pd.DataFrame(rows, columns=["k", "pixhash", "mode", "shape"])
    assert h.k.is_unique
    h.to_csv(WORK / f"hashes_{ds}.csv.gz", index=False)
    print(ds, len(h), h["mode"].value_counts().to_dict())


def arms():
    k = pd.read_csv(SPL / "kermany.csv.gz", low_memory=False)
    k["k"], k["g"] = k.version + ":" + k.key, k.pid.astype("Int64").astype(str)
    v2, v3 = k[k.version == "v2"], k[(k.version == "v3") & (k.label != "CXR")]
    i = pd.read_csv(SPL / "isic2019.csv.gz")
    i["k"], i["g"] = "isic:" + i.image, i.group.astype(str)
    out = {}
    for f in ("b0", "b1"):
        c = f"b_f{f[1]}"
        out[f"kermany_{f}"] = ({r: v3[v3[c] == r] for r in ("train", "val", "seen", "unseen")}, v3[v3.label == "DRUSEN"])
        out[f"isic2019_{f}"] = ({r: i[i[c] == r] for r in ("train", "val", "seen", "unseen")}, i[i.role_std == "ood"])
    out["kermany_std"] = ({**{s: v2[v2.role_std == r] for s, r in (("train", "train"), ("val", "val"), ("seen", "test_seen"),
                                                                     ("unseen", "test_unseen"))},
                           "unseen_extra": k[k.role_std == "unseen_extra"]}, v2[v2.role_std == "ood"])
    out["isic2019_std"] = ({s: i[i.role_std == r] for s, r in (("train", "train"), ("val", "val"), ("seen", "test_seen"),
                                                              ("unseen", "test_unseen"))}, i[i.role_std == "ood"])
    return out


def do_audit():
    H = pd.concat([pd.read_csv(WORK / f"hashes_{ds}.csv.gz") for ds in ("kermany", "isic2019")]).set_index("k").pixhash
    res = {}
    for name, (ids, ood) in arms().items():
        assert ood.k.isin(H.index).all() and all(s.k.isin(H.index).all() for s in ids.values()), name
        og, oh = ood.g.values, H.loc[ood.k].values
        r = {"group_unit": "patient ID" if name.startswith("kermany") else "lesion_id, else image (no patient ID in ISIC 2019)",
             "n_ood_images": len(ood), "n_ood_groups": int(len(set(og))),
             "id_sets": {s: {"images": len(x), "groups": int(x.g.nunique())} for s, x in ids.items()}, "by_set": {}}
        any_g = set().union(*(set(x.g) for x in ids.values()))
        for s, x in ids.items():
            sg, sh = set(x.g), H.loc[x.k]
            cnt_s = sh.value_counts()
            cnt_o = pd.Series(oh).value_counts()
            common = cnt_o.index.intersection(cnt_s.index)
            same_g = 0
            if len(common):
                m = pd.DataFrame({"h": sh.values, "g": x.g.values})
                o = pd.DataFrame({"h": oh, "g_ood": og})
                m = m[m.h.isin(common)].merge(o[o.h.isin(common)], on="h")
                same_g = int((m.g == m.g_ood).sum())
            pairs = int((cnt_o[common] * cnt_s[common]).sum())
            r["by_set"][s] = {"ood_groups_in_set": int(len(set(og) & sg)),
                              "ood_images_from_set_groups": int(np.isin(og, list(sg)).sum()),
                              "pixel_identical_pairs": pairs, "pixel_identical_pairs_same_group": same_g,
                              "pixel_identical_pairs_cross_group": pairs - same_g,
                              "ood_images_with_identical_in_set": int(np.isin(oh, common).sum()),
                              "set_images_with_identical_in_ood": int(sh.isin(common).sum())}
        clean = ~np.isin(og, list(any_g))
        r["ood_groups_in_any_id_set"] = int(len(set(og) & any_g))
        r["ood_images_from_any_id_group"] = int((~clean).sum())
        r["clean_ood_images"], r["clean_ood_groups"] = int(clean.sum()), int(len(set(og[clean])))
        ch = oh[clean]
        id_h = set(pd.concat([H.loc[x.k] for x in ids.values()]))
        r["clean_ood_images_with_identical_in_any_id_set"] = int(np.isin(ch, list(id_h)).sum())
        r["ood_within_set_identical_extra_copies"] = int(len(oh) - len(set(oh)))
        res[name] = r
        print(name, r["n_ood_images"], r["ood_images_from_any_id_group"], r["clean_ood_images"],
              {s: v["pixel_identical_pairs"] for s, v in r["by_set"].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    do_hash(sys.argv[2]) if sys.argv[1] == "hash" else do_audit()
