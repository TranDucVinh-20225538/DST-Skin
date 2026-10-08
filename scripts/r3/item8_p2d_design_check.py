#!/usr/bin/env python3
"""R3 item 8 / P2-d design checks on the Kvasir-Capsule label files (no images, no model, no Delta).

1. Video / frame parsing, two independent ways on the official split files: (a) str.rsplit('_', 1); (b) a strict
   regex ^([0-9a-f]{16})_([0-9]+)\\.jpg$. With --metadata, (c) the video_id / frame_number columns of metadata.csv
   from the labelled-image zip. Reports the number of videos listed in both official folds under each parser.
2. Temporal-gap feasibility: for each held-out-class candidate, build the P2-d sets (OOD-class videos removed; 20%
   of videos unseen; 10% of the rest ckpt; frame-level random 70 / 15 / 15 stratified on class; default_rng(0)) and
   report the share of fit frames within +/- GAP frames of a test (seen) frame of the same video, i.e. the share
   the gap variant removes, and the fit frames / videos left.
Writes results/r3/8/p2d/design_check.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KC = "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_{}.csv"
RX = re.compile(r"^([0-9a-f]{16})_([0-9]+)\.jpg$")
GAP = 25


def load():
    d = pd.concat([pd.read_csv(io.BytesIO(urllib.request.urlopen(KC.format(s), timeout=120).read())).assign(fold=s)
                   for s in (0, 1)], ignore_index=True)
    d["video_a"] = d.filename.str.rsplit("_", n=1).str[0]
    d["frame_a"] = d.filename.str.rsplit("_", n=1).str[1].str.replace(".jpg", "", regex=False).astype(int)
    m = d.filename.map(RX.match)
    assert m.notna().all(), d.filename[m.isna()].head().tolist()
    d["video_b"] = m.map(lambda x: x.group(1))
    d["frame_b"] = m.map(lambda x: int(x.group(2)))
    return d


def both_folds(d, col):
    t = d.groupby([col, "fold"]).size().unstack(fill_value=0)
    b = t[(t[0] > 0) & (t[1] > 0)]
    return {"videos": int(t.shape[0]), "videos_in_both_folds": int(len(b)), "frames_in_those_videos": int(b.to_numpy().sum()),
            "minority_side_frames": int(b.min(axis=1).sum()), "ids": sorted(b.index.tolist())}


def sets(u, ood_class, rng_seed=0):
    vs_ood = set(u.loc[u.label == ood_class, "video"])
    x = u[~u.video.isin(vs_ood)].copy()
    vids = np.array(sorted(x.video.unique()))
    major = x.groupby("video").label.agg(lambda s: s.value_counts().index[0])
    rng = np.random.default_rng(rng_seed)
    unseen = set()
    for _, g in major.groupby(major):
        g = np.array(sorted(g.index))
        unseen |= set(g[rng.permutation(len(g))[: int(round(0.2 * len(g)))]])
    rest = np.array([v for v in vids if v not in unseen])
    ckpt = set(rest[rng.permutation(len(rest))[: int(round(0.1 * len(rest)))]])
    y = x[~x.video.isin(unseen | ckpt)].copy()
    role = np.empty(len(y), dtype=object)
    for _, idx in y.groupby("label").indices.items():
        p = rng.permutation(idx)
        n_tr, n_va = int(round(0.70 * len(p))), int(round(0.15 * len(p)))
        role[p[:n_tr]], role[p[n_tr:n_tr + n_va]], role[p[n_tr + n_va:]] = "fit", "val", "test"
    y["role"] = role
    return y, len(vs_ood), len(unseen), len(ckpt), int(x[x.video.isin(unseen)].shape[0])


def gap_share(y, gap=GAP):
    removed = 0
    fit = y[y.role == "fit"]
    for v, g in fit.groupby("video"):
        t = np.sort(y.loc[(y.video == v) & (y.role == "test"), "frame"].to_numpy())
        if not len(t):
            continue
        f = g.frame.to_numpy()
        i = np.searchsorted(t, f)
        lo = np.abs(f - t[np.clip(i - 1, 0, len(t) - 1)])
        hi = np.abs(t[np.clip(i, 0, len(t) - 1)] - f)
        removed += int((np.minimum(lo, hi) <= gap).sum())
    left = fit.shape[0] - removed
    return removed, left


def segment_roles(y, seed=0):
    """Segment-level split: contiguous same-label runs (frame step 1) go wholly to fit / val / test, 70 / 15 / 15 by
    frames, stratified on class, default_rng(seed)."""
    y = y.sort_values(["video", "frame"]).copy()
    y["seg"] = ((y.video != y.video.shift()) | (y.frame.diff() != 1) | (y.label != y.label.shift())).cumsum()
    rng, role = np.random.default_rng(seed), {}
    for _, g in y.groupby("label"):
        s = g.groupby("seg").size()
        s = s.iloc[rng.permutation(len(s))]
        for k, cf in (s.cumsum() / s.sum()).items():
            role[k] = "fit" if cf <= 0.70 else ("val" if cf <= 0.85 else "test")
    y["role"] = y.seg.map(role)
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None, help="metadata.csv from the labelled-image zip (third parser)")
    a = ap.parse_args()
    d = load()
    out = {"parse": {"rows": len(d), "a_equals_b_video": bool((d.video_a == d.video_b).all()),
                     "a_equals_b_frame": bool((d.frame_a == d.frame_b).all()),
                     "a_rsplit": both_folds(d, "video_a"), "b_regex": both_folds(d, "video_b")}}
    if a.metadata:
        m = pd.read_csv(a.metadata, sep=None, engine="python")
        m.columns = [c.strip().lower() for c in m.columns]
        out["metadata_columns"] = list(m.columns)
        key = m.filename.str.replace(r"\.(png|jpg)$", "", regex=True)
        mm = dict(zip(key, zip(m.video_id.astype(str), m.frame_number.astype(int))))
        k = d.filename.str.replace(".jpg", "", regex=False)
        hit = k.isin(mm.keys())
        d.loc[hit, "video_c"] = [mm[x][0] for x in k[hit]]
        d.loc[hit, "frame_c"] = [mm[x][1] for x in k[hit]]
        out["parse"]["c_metadata"] = {"matched_rows": int(hit.sum()),
                                      "video_equal_a": bool((d.loc[hit, "video_c"] == d.loc[hit, "video_a"]).all()),
                                      "frame_equal_a": bool((d.loc[hit, "frame_c"].astype(int) == d.loc[hit, "frame_a"]).all()),
                                      **both_folds(d[hit], "video_c")}
    multi = d.filename[d.filename.duplicated(keep=False)].unique()
    u = d[~d.filename.isin(multi)].rename(columns={"video_a": "video", "frame_a": "frame"})
    gaps = {}
    for c in ["Angiectasia", "Erosion", "Reduced Mucosal View", "Ulcer", "Foreign Bodies"]:
        y, n_ood_v, n_unseen_v, n_ckpt_v, n_unseen_f = sets(u, c)
        rem, left = gap_share(y)
        fit = y[y.role == "fit"]
        left_v = fit.groupby("video").size()
        gaps[c] = {"ood_videos": n_ood_v, "ood_frames": int((u.label == c).sum()), "meets_ge5_ood_videos": n_ood_v >= 5,
                   "unseen_videos": n_unseen_v, "unseen_frames": n_unseen_f, "ckpt_videos": n_ckpt_v,
                   "fit_frames": int(fit.shape[0]), "fit_videos": int(fit.video.nunique()),
                   "seen_frames": int((y.role == "test").sum()),
                   "gap_removed_frames": rem, "gap_removed_share": round(rem / max(len(fit), 1), 4),
                   "fit_frames_left_after_gap": left,
                   "id_classes_with_ge2_videos": int((u[~u.video.isin(set(u.loc[u.label == c, 'video']))]
                                                       .groupby("label").video.nunique() >= 2).sum())}
    out["gap"] = {"gap_frames": GAP, "frame_level_split": gaps}
    sweep, seg = {}, {}
    y, *_ = sets(u, "Angiectasia")
    for gp in (1, 2, 5, 10, 25):
        rem, left = gap_share(y, gp)
        sweep[gp] = {"removed_share": round(rem / int((y.role == "fit").sum()), 4), "fit_left": left}
    out["gap"]["frame_level_sweep_angiectasia"] = sweep
    runs = []
    for _, g in u.groupby("video"):
        f = np.sort(g.frame.unique())
        runs += list(np.diff(np.r_[-1, np.where(np.diff(f) > 1)[0], len(f) - 1]))
    runs = np.array(runs)
    out["gap"]["contiguous_runs"] = {"n": len(runs), "median_len": int(np.median(runs)), "max_len": int(runs.max()),
                                     "share_frames_in_runs_ge50": round(float(runs[runs >= 50].sum() / runs.sum()), 3)}
    for c in ["Angiectasia", "Erosion"]:
        y, *_ = sets(u, c)
        z = segment_roles(y)
        rem, left = gap_share(z)
        fit, te = z[z.role == "fit"], z[z.role == "test"]
        seg[c] = {"segments": int(z.seg.nunique()), "fit_frames": len(fit), "seen_frames": len(te),
                  "gap_removed_share": round(rem / len(fit), 4), "fit_frames_left_after_gap": left,
                  "seen_frames_whose_video_has_fit_frames": round(float(te.video.isin(set(fit.video)).mean()), 3)}
    out["gap"]["segment_level_split"] = seg
    p = REPO / "results/r3/8/p2d/design_check.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "ids"} if isinstance(v, dict) else v
                      for k, v in out["parse"].items()}, indent=1))
    for c, g in gaps.items():
        print(c, g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
