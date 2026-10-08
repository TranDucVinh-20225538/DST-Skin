#!/usr/bin/env python3
"""R3 item 1 report: results/r3/1/REPORT.md and results/r3/1/paper_ci_v2.csv.

Rows: recompute_paper_ci.py --protocol paper_2fold with the Track A scorers (mahalanobis_l2, knn_mean_cosine) from
~/r3work/item1/out_v2: medical cells (list_v2_med.txt, seen = strict), ISIC / Kermany cells also with seen = tracka
(list_v2_seentracka.txt, suffix _seentracka), Camelyon (list_camelyon.txt, suffixes _maha / _knnmc). Match table from
results/r3/1/match_table.md (summarize_match.py). Usage: item1_report.py [<commit>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
W = Path.home() / "r3work/item1"
OUT = REPO / "results/r3/1"
PRIORITY = ("uni", "virchow2", "dermamnist", "kermany", "isic2019")


def lines(p):
    return [x.strip() for x in (W / p).read_text().splitlines() if x.strip()]


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "uncommitted"
    cells = pd.concat([pd.read_csv(W / f"cells_{p}.csv") for p in ("med", "medfm", "camelyon")]).set_index("cell")
    want = [(c, "", "strict") for c in lines("list_v2_med.txt")]
    want += [(c, "_seentracka", "tracka") for c in lines("list_v2_seentracka.txt")]
    want += [(c, s, "strict") for c in lines("list_camelyon.txt") for s in ("_maha", "_knnmc")]
    rows, missing = [], []
    for c, suf, seen in want:
        p = W / "out_v2" / f"{c}{suf}" / "paper_ci.csv"
        if not p.exists() or p.stat().st_size == 0:
            missing.append(f"{c}{suf}")
            continue
        d = pd.read_csv(p)
        assert (d["seen"] == seen).all(), (p, d["seen"].unique())
        m = cells.loc[c]
        for _, r in d.iterrows():
            rows.append(dict(cell=c, dataset=m.dataset, model=m.model, seed=int(m.seed), kind=m.kind, scorer=r.scorer, seen=seen,
                             delta=r.delta, trackA_delta=m[f"trackA_delta_{'knn' if 'knn' in r.scorer else 'mahalanobis'}"],
                             old_lo=r.old_bootstrap_lo, old_hi=r.old_bootstrap_hi, new_lo=r.new_jackknife_lo,
                             new_hi=r.new_jackknife_hi, status_thr=r.status_thr, status_gt0=r.status_gt0,
                             n_groups_train=int(r.n_groups_train), K=int(r.K), n_groups_fit=f"{int(r.n_groups_train)}/{int(r.K)}",
                             d=int(m.d), id_acc=m.id_acc, n_id_eval=int(r.n_id_eval), n_id_eval_orphan=int(r.n_id_eval_orphan),
                             n_ood=int(r.n_ood)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "paper_ci_v2.csv", index=False)

    def tab(df):
        h = ("| cell | scorer | seen | Δ | Track A Δ | old CI (bootstrap) | new CI (jackknife) | status vs 0.02 | status vs 0 | "
             "n_groups_fit (per fold) | K | d | ID acc | orphan seen |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        b = [f"| {r.cell} | {r.scorer} | {r.seen} | {r.delta:.4f} | {r.trackA_delta:.4f} | [{r.old_lo:.4f}, {r.old_hi:.4f}] | "
             f"[{r.new_lo:.4f}, {r.new_hi:.4f}] | {r.status_thr} | {r.status_gt0} | {r.n_groups_fit} | {r.K} | {r.d} | "
             f"{r.id_acc:.4f} | {r.n_id_eval_orphan} |" for r in df.itertuples()]
        return [h] + b

    st = T[T.seen == "strict"]
    cnt = st.groupby("status_thr").size().to_dict()
    L = ["# R3 item 1: paper CIs on real features (paper_2fold, Track A scorers)", "", f"Commit: {commit}", "",
         "Verdict: vs 0.02 (seen = strict): " + ", ".join(f"{k} {cnt.get(k, 0)}" for k in
                                                   ("pass_both", "pass_old_only", "pass_new_only", "fail_both"))
         + f" of {len(st)} rows; not for the manuscript until item 2 passes.", ""]
    if missing:
        L += [f"Not run / no output ({len(missing)}): " + ", ".join(missing), ""]
    L += ["## Stop-check: package point estimate vs Track A Δ_fit (tolerance 0.001)", ""]
    L += (OUT / "match_table.md").read_text().strip().splitlines()
    L += ["", "## pass_old_only rows", ""]
    po = T[(T.status_thr == "pass_old_only") | (T.status_gt0 == "pass_old_only")]
    L += tab(po) if len(po) else ["None."]
    L += ["", "## Priority cells (UNI, Virchow2, DermaMNIST, Kermany, ISIC 2019)", ""]
    pr = T[T.cell.str.contains("|".join(PRIORITY))]
    L += tab(pr)
    L += ["", "## All rows", ""] + tab(T.sort_values(["dataset", "cell", "scorer", "seen"]))
    L += ["", "## Caveats", "",
          "1. Scorers are the Track A definitions (mahalanobis_l2 = Ledoit-Wolf float64 Mahalanobis; knn_mean_cosine = mean "
          "cosine distance to 50 neighbours). Package-kNN numbers are not used. ViM is not reported: Track A ViM is not "
          "reproduced within 0.001 (match table).",
          "2. seen = tracka rows (ISIC 2019, Kermany) count orphan seen groups (no training image after the val carve-out) as "
          "jackknife units, as Track A does; coverage (item 2) is verified under the strict definition only.",
          "3. Camelyon knn_mean_cosine ran on GPU after passing the strict equality check against sklearn (CPU); all other rows CPU.",
          "4. Technical rerun: the first submission of the medical cells failed at start (a GPU script path leaked into the "
          "CPU jobs' environment); rerun unchanged on CPU (jobs 64813 / 64814).",
          "5. n_groups_fit is per fold: the scorer is fit on one of K halves of the n_groups_train training groups."]
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
