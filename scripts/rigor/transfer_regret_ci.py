#!/usr/bin/env python
"""Post-hoc (not in the precommit): is the cross-arch transfer regret of H16 distinguishable from 0?

Camelyon17, kind cross_arch, families nonfeature and all. Two readings per root:
  mean_ci : mean regret over all ordered arch pairs and seeds, 95% percentile CI from a two-way
            cluster bootstrap that resamples archs and seeds (model-level uncertainty).
            Regret is >= 0 by construction, so this CI excludes 0 whenever any pair is > 0.
  pairs   : per (seed, src, tgt) pair, the AUROC loss on the tgt cell of the src-chosen score vs
            the tgt-best score, with a patient-clustered DeLong SE (patches of one patient are
            not independent); one-sided p, Holm over all pairs of the family.

Usage: python scripts/rigor/transfer_regret_ci.py --root ROOT --label NAME [--out DIR]
Writes DIR/transfer_regret_ci.csv (one row per family, appended/replaced by label) and
DIR/transfer_regret_ci_pairs_<label>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rigor import common as C  # noqa: E402
from scripts.rigor.persample import clusters_for, rankdata  # noqa: E402
from scripts.rigor.transfer_regret import FAMILIES  # noqa: E402

DOMAIN = "camelyon17"
KEY = {v: k for k, v in C.DISPLAY.items()}


def components(id_s, ood_s):
    p = np.asarray(id_s, np.float32).astype(np.float64)
    q = np.asarray(ood_s, np.float32).astype(np.float64)
    m, n = len(p), len(q)
    tx, ty, tz = rankdata(p), rankdata(q), rankdata(np.concatenate([p, q]))
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    return auc, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m


def clustered_var_of_mean(x, codes):
    if codes is None:
        return float(np.var(x, ddof=1) / len(x))
    k = int(codes.max()) + 1
    s = np.bincount(codes, weights=x - x.mean(), minlength=k)
    return float((s ** 2).sum() / len(x) ** 2 * k / (k - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.REPO))
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=str(C.default_reports(C.REPO) / "numerical_stability"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()
    root, out = Path(args.root), C.ensure_dir(Path(args.out))
    archs, seeds = list(C.ARCHS), list(C.SEEDS)
    cube = C.load_cube(root, DOMAIN, archs, seeds)  # (A, S, M) AUROC, same source as transfer_regret.py
    rng = np.random.default_rng(args.rng)

    comp, codes = {}, None
    for ai, a in enumerate(archs):
        for s in seeds:
            z = np.load(C.score_cache_path(C.default_out(root), DOMAIN, a, s))
            if codes is None:
                idc, oc, _, note = clusters_for(DOMAIN, C.REPO, len(z["id_labels"]), len(z["ood_labels"]),
                                                z["id_labels"], z["ood_labels"])
                codes = (idc, oc)
                print(note)
            comp[(a, s)] = {m: components(z["id_" + KEY[m]], z["ood_" + KEY[m]]) for m in C.METHODS_ORDER}

    summary, pair_rows = [], []
    for fam, ms in FAMILIES.items():
        if fam == "feature":
            continue
        idx = [C.METHODS_ORDER.index(m) for m in ms if m in C.METHODS_ORDER]
        R = np.full((len(seeds), len(archs), len(archs)), np.nan)
        rows = []
        for si, s in enumerate(seeds):
            for a in range(len(archs)):
                for b in range(len(archs)):
                    if a == b:
                        continue
                    src, tgt = cube[a, si, idx], cube[b, si, idx]
                    m_src = C.METHODS_ORDER[idx[int(np.argmax(src))]]
                    m_best = C.METHODS_ORDER[idx[int(np.argmax(tgt))]]
                    R[si, a, b] = float(np.max(tgt) - tgt[int(np.argmax(src))])
                    c = comp[(archs[b], s)]
                    a1, v01a, v10a = c[m_best]
                    a2, v01b, v10b = c[m_src]
                    d = a1 - a2
                    var = clustered_var_of_mean(v01a - v01b, codes[0]) + clustered_var_of_mean(v10a - v10b, codes[1])
                    se = float(np.sqrt(var)) if var > 0 else 0.0
                    p = 1.0 if d <= 0 or se == 0 else C.norm_sf(d / se)
                    rows.append({"label": args.label, "family": fam, "seed": s, "src": archs[a], "tgt": archs[b],
                                 "score_src_choice": m_src, "score_tgt_best": m_best, "regret": R[si, a, b],
                                 "auroc_diff_cache": d, "se_patient_clustered": se,
                                 "ci95_lo_one_sided": d - 1.6448536269514722 * se, "p_one_sided": p})
        pr = pd.DataFrame(rows)
        pr["p_holm"] = C.holm(pr.p_one_sided.to_numpy())
        pair_rows.append(pr)

        A, S = len(archs), len(seeds)
        means, fracs = np.empty(args.B), np.empty(args.B)
        for r in range(args.B):
            aa = rng.integers(0, A, A)
            ss = rng.integers(0, S, S)
            sub = R[np.ix_(ss, aa, aa)]
            keep = aa[:, None] != aa[None, :]
            vals = sub[:, keep]
            means[r] = np.nanmean(vals)
            fracs[r] = np.nanmean(vals > 0.05)
        lo, hi = C.percentile_ci(means)
        flo, fhi = C.percentile_ci(fracs)
        v = R[~np.isnan(R)]
        summary.append({"label": args.label, "family": fam, "n_pairs": int(v.size),
                        "mean_regret": float(v.mean()), "mean_ci95_lo": lo, "mean_ci95_hi": hi,
                        "median_regret": float(np.median(v)), "max_regret": float(v.max()),
                        "frac_gt_0.05": float((v > 0.05).mean()), "frac_gt_0.05_ci95_lo": flo,
                        "frac_gt_0.05_ci95_hi": fhi, "frac_zero": float((v == 0).mean()),
                        "n_pairs_loss_sig_unadj": int((pr.p_one_sided < 0.05).sum()),
                        "n_pairs_loss_sig_holm": int((pr.p_holm < 0.05).sum()),
                        "n_pairs_loss_sig_holm_and_gt_0.05": int(((pr.p_holm < 0.05) & (pr.regret > 0.05)).sum()),
                        "max_abs_regret_vs_cache_diff": float(np.abs(pr.regret - pr.auroc_diff_cache).max()),
                        "R50_ConvNeXt_s42": float(pr[(pr.src == "resnet50") & (pr.tgt == "convnext_tiny")
                                                     & (pr.seed == 42)].regret.iloc[0])})

    pd.concat(pair_rows).to_csv(out / ("transfer_regret_ci_pairs_%s.csv" % args.label), index=False)
    f = out / "transfer_regret_ci.csv"
    new = pd.DataFrame(summary)
    if f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old[old.label != args.label], new])
    new.to_csv(f, index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
