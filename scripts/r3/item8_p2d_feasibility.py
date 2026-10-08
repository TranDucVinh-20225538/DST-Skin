#!/usr/bin/env python3
"""R3 item 8 / P2-d feasibility from public metadata only (no image download, no terms accepted).

Kvasir-Capsule: official two-fold split files (github.com/simula/kvasir-capsule, official_splits/split_{0,1}.csv;
  filename = {video_id}_{frame}.jpg) -> frames per video, class balance, video overlap between the two folds,
  held-out-class options (OOD class frames' videos removed from ID).
Galar: Galar_labels_and_metadata.7z (2.3 MB, figshare+ 25304616) -> capsule system and frame counts per video.
Brain tumor (Cheng): figshare API record 1512427 (licence, files, sizes); README fields are quoted in the precommit.
Writes results/r3/8/p2d/feasibility.json. Scratch in a temporary directory, removed on exit.
"""

from __future__ import annotations

import io
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/r3/8/p2d/feasibility.json"
KC = "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_{}.csv"
GALAR_API, BRAIN_API = "https://api.figshare.com/v2/articles/25304616", "https://api.figshare.com/v2/articles/1512427"
PATHO = ["ulcer", "polyp", "active bleeding", "blood", "erythema", "erosion", "angiectasia", "foreign body",
         "lymphangioectasis"]


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def q(x):
    return {k: int(np.quantile(x, p)) for k, p in (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("max", 1))}


def kvasir():
    d = pd.concat([pd.read_csv(io.BytesIO(get(KC.format(s)))).assign(fold=s) for s in (0, 1)])
    d["video"] = d.filename.str.rsplit("_", n=1).str[0]
    d["frame"] = d.filename.str.rsplit("_", n=1).str[1].str[:-4].astype(int)
    multi = d[d.filename.duplicated(keep=False)]
    u = d[~d.filename.isin(multi.filename)]
    per_v = u.groupby("video").size()
    vf = u.groupby(["video", "fold"]).size().unstack(fill_value=0)
    both = vf[(vf[0] > 0) & (vf[1] > 0)]
    s = set(zip(u.video, u.frame))
    adj = float(np.mean([(v, f - 1) in s or (v, f + 1) in s for v, f in zip(u.video, u.frame)]))
    held = {}
    for c in ["Ulcer", "Foreign Bodies", "Angiectasia", "Lymphangiectasia", "Blood", "Erosion", "Erythematous"]:
        vs = set(u[u.label == c].video)
        rest = u[~u.video.isin(vs)]
        held[c] = {"ood_frames": int((u.label == c).sum()), "ood_videos": len(vs), "id_frames_left": len(rest),
                   "id_videos_left": int(rest.video.nunique()),
                   "id_classes_with_ge2_videos": int((rest.groupby("label").video.nunique() >= 2).sum())}
    return {"rows": len(d), "multi_label_files": int(multi.filename.nunique()), "single_label_frames": len(u),
            "videos": int(u.video.nunique()), "frames_per_video": q(per_v.to_numpy()),
            "frames_per_video_top5_share": round(float(per_v.nlargest(5).sum() / len(u)), 3),
            "class_counts": u.label.value_counts().to_dict(),
            "videos_per_class": u.groupby("label").video.nunique().sort_values().to_dict(),
            "official_folds": {"videos_per_fold": u.groupby("fold").video.nunique().to_dict(),
                               "videos_in_both_folds": len(both), "frames_in_those_videos": int(both.to_numpy().sum()),
                               "minority_side_frames": int(both.min(axis=1).sum())},
            "frac_frames_with_adjacent_labelled_frame": round(adj, 3), "held_out_class_options": held}


def galar(tmp):
    meta = json.loads(get(GALAR_API))
    fid = next(f["id"] for f in meta["files"] if f["name"].startswith("Galar_labels"))
    (tmp / "lm.7z").write_bytes(get(f"https://ndownloader.figshare.com/files/{fid}"))
    subprocess.run(["7z", "x", "-y", f"-o{tmp}", str(tmp / "lm.7z")], check=True, stdout=subprocess.DEVNULL)
    m = pd.read_csv(tmp / "metadata.csv")
    m.columns = [c.strip() for c in m.columns]
    rows = []
    for f in (tmp / "Labels").glob("*.csv"):
        x = pd.read_csv(f)
        si = x["small intestine"] == 1
        rows.append({"video": int(f.stem), "frames": len(x), "small_intestine": int(si.sum()),
                     "small_intestine_pathology": int((x.loc[si, PATHO].sum(axis=1) > 0).sum())})
    r = pd.DataFrame(rows).merge(m, left_on="video", right_on="File Name")
    by = r.groupby("Capsule System").agg(videos=("video", "size"), frames=("frames", "sum"),
                                         small_intestine=("small_intestine", "sum"),
                                         small_intestine_pathology=("small_intestine_pathology", "sum"))
    return {"licence": meta["license"]["name"], "total_bytes": meta["size"],
            "files": {f["name"]: f["size"] for f in meta["files"]},
            "by_capsule_system": by.astype(int).to_dict(orient="index"),
            "olympus_videos": sorted(r.loc[r["Capsule System"].str.contains("Olympus"), "video"].tolist())}


def brain():
    meta = json.loads(get(BRAIN_API))
    return {"licence": meta["license"]["name"], "doi": meta["doi"], "total_bytes": meta["size"],
            "files": {f["name"]: f["size"] for f in meta["files"]}}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="r3p2d_") as t:
        res = {"kvasir_capsule": kvasir(), "galar": galar(Path(t)), "brain_tumor_cheng": brain()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["kvasir_capsule"]["official_folds"]), res["galar"]["by_capsule_system"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
