#!/usr/bin/env python3
"""Rigor pack, stage CSV (CPU, seconds-minutes, no torch needed).

Reads ONLY tracked score CSVs + the frozen ranks file. Never rewrites official W.

Analyses (decisions/decision_precommit_rigor_pack.md, H1, H2, H4, H5, H7a, H8a, H12):
  1. Reproduction checks against published numbers (0.694, 0.791, cross-seed W, 4-anchor W).
  2. Rater-count-matched W: cross-arch W over all C(8,5)=56 five-arch subsets, per seed 42-46,
     vs cross-seed W (k=5 seeds) for ALL 8 archs (seed 43-46 CSVs exist for all 8).
  3. Permutation null (10k, within-rater shuffle) + Friedman for every W.
  4. Architecture-label permutation test: are 5 runs of the same arch more concordant than
     5 random runs from the 40-run pool? (the direct "arch vs seed" test)
  5. Nested variance decomposition (method x arch x seed(arch)) on AUROC and on ranks,
     with F test and arch-label permutation p. Also a crossed-seed sensitivity.
  6. Feature-space rank-1 count over every arch x seed cell.
  7. W on FPR95 (from the same CSVs) as an alternative metric.
  8. MSP-jump tests per arch over 5 seeds (t-test vs 0 and vs 0.15) for multiple testing.
Repeated for skin and MIDOG (8 archs x seeds 42-46, if CSVs present).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def cross_seed_w(cube: np.ndarray, ai: int):
    mat = cube[ai]
    if np.isnan(mat).any():
        return float("nan"), None
    r = C.ranks_rows(mat)
    return C.kendall_w(r), r


def cross_arch_w(cube: np.ndarray, si: int, arch_idx):
    mat = cube[list(arch_idx), si]
    if np.isnan(mat).any():
        return float("nan"), None
    r = C.ranks_rows(mat)
    return C.kendall_w(r), r


def nested_anova(y: np.ndarray) -> dict:
    """y: (A, S, M). Seed nested in arch. Returns SS/df/MS and variance components."""
    A, S, M = y.shape
    g = y.mean()
    ym = y.mean(axis=(0, 1))  # M
    ya = y.mean(axis=(1, 2))  # A
    yas = y.mean(axis=2)  # A,S
    yma = y.mean(axis=1)  # A,M
    ss = {
        "method": A * S * np.sum((ym - g) ** 2),
        "arch": M * S * np.sum((ya - g) ** 2),
        "seed(arch)": M * np.sum((yas - ya[:, None]) ** 2),
        "method:arch": S * np.sum((yma - ym[None, :] - ya[:, None] + g) ** 2),
        "method:seed(arch)": np.sum((y - yma[:, None, :] - yas[:, :, None] + ya[:, None, None]) ** 2),
    }
    df = {"method": M - 1, "arch": A - 1, "seed(arch)": A * (S - 1), "method:arch": (M - 1) * (A - 1),
          "method:seed(arch)": (M - 1) * A * (S - 1)}
    ms = {k: ss[k] / df[k] if df[k] > 0 else np.nan for k in ss}
    tot = np.sum((y - g) ** 2)
    f = ms["method:arch"] / ms["method:seed(arch)"] if ms["method:seed(arch)"] > 0 else np.nan
    try:
        from scipy.stats import f as fdist

        p = float(fdist.sf(f, df["method:arch"], df["method:seed(arch)"]))
    except Exception:
        p = float("nan")
    # method-of-moments variance components for the ranking-relevant (interaction) part
    s2_ma = max(0.0, (ms["method:arch"] - ms["method:seed(arch)"]) / S)
    s2_ms = ms["method:seed(arch)"]
    out = {"ss_total": tot, "F_methodxarch_vs_methodxseed": f, "p_F": p,
           "share_ss_interaction_arch": ss["method:arch"] / (ss["method:arch"] + ss["method:seed(arch)"]),
           "varcomp_methodxarch": s2_ma, "varcomp_methodxseed": s2_ms,
           "share_varcomp_arch": s2_ma / (s2_ma + s2_ms) if (s2_ma + s2_ms) > 0 else np.nan}
    for k in ss:
        out["ss_" + k] = ss[k]
        out["df_" + k] = df[k]
        out["frac_total_" + k] = ss[k] / tot if tot > 0 else np.nan
    return out


def crossed_anova_fracs(y: np.ndarray) -> dict:
    """Sensitivity: treat seed number as a crossed factor (A x S x M, no replication)."""
    A, S, M = y.shape
    g = y.mean()
    ya, ys, ym = y.mean(axis=(1, 2)), y.mean(axis=(0, 2)), y.mean(axis=(0, 1))
    yas, yam, ysm = y.mean(axis=2), y.mean(axis=1), y.mean(axis=0)
    ss = {
        "arch": S * M * np.sum((ya - g) ** 2),
        "seed": A * M * np.sum((ys - g) ** 2),
        "method": A * S * np.sum((ym - g) ** 2),
        "arch:seed": M * np.sum((yas - ya[:, None] - ys[None, :] + g) ** 2),
        "arch:method": S * np.sum((yam - ya[:, None] - ym[None, :] + g) ** 2),
        "seed:method": A * np.sum((ysm - ys[:, None] - ym[None, :] + g) ** 2),
    }
    tot = np.sum((y - g) ** 2)
    ss["arch:seed:method"] = tot - sum(ss.values())
    return {"crossed_frac_" + k: v / tot for k, v in ss.items()}


def arch_label_perm(units: np.ndarray, A: int, S: int, n_perm: int, rng, stat_fn):
    """units: (A*S, ...) per-run data, grouped arch-major. Shuffle run->arch labels."""
    obs = stat_fn(units.reshape((A, S) + units.shape[1:]))
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(A * S)
        null[t] = stat_fn(units[perm].reshape((A, S) + units.shape[1:]))
    return obs, null


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.REPO), help="repo root holding outputs/reports (tracked CSVs)")
    ap.add_argument("--reports", default=None, help="output dir (default outputs/reports/rigor_pack/w_csv)")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--domains", nargs="+", default=list(C.DOMAINS))
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.reports) if args.reports else C.default_reports(root) / "w_csv"
    C.ensure_dir(out)
    rng = np.random.default_rng(args.seed)
    archs = list(C.ARCHS)
    seeds = list(C.SEEDS)
    M = len(C.METHODS_ORDER)

    repro, subsets_rows, cs_rows, summ_rows, perm_rows, alp_rows = [], [], [], [], [], []
    anova_rows, rank1_rows, fpr_rows, jump_rows, pfam = [], [], [], [], []
    log = ["Rigor pack / w_from_csv  " + str(C.env_info()), ""]

    for domain in args.domains:
        cube = C.load_cube(root, domain, archs, seeds, "AUROC")  # (A,S,M)
        avail = ~np.isnan(cube).any(axis=2)
        log.append("[%s] complete arch x seed cells: %d / %d" % (domain, int(avail.sum()), avail.size))
        for ai, a in enumerate(archs):
            miss = [s for si, s in enumerate(seeds) if not avail[ai, si]]
            if miss:
                log.append("  missing %s seeds %s" % (a, miss))
        if avail.sum() == 0:
            continue

        # ---- 1. reproduction checks (published numbers) ----
        if domain in ("camelyon17", "skin_isic_pad") and avail[:, 0].all():
            w42, _ = cross_arch_w(cube, 0, range(len(archs)))
            key = "official_w_camelyon" if domain == "camelyon17" else "official_w_skin"
            pub = C.PUBLISHED[key]
            repro.append({"check": "%s official W seed42 n=8" % domain, "published": pub, "recomputed": w42,
                          "pass": abs(round(w42, 3) - pub) < 1e-9})
        if domain == "camelyon17":
            for a, pub in C.PUBLISHED["cross_seed_w"].items():
                w, _ = cross_seed_w(cube, archs.index(a))
                repro.append({"check": "cross-seed W %s" % a, "published": pub, "recomputed": w,
                              "pass": abs(round(w, 3) - pub) < 1e-9})
            for s, pub in C.PUBLISHED["anchor4_cross_arch_w"].items():
                w, _ = cross_arch_w(cube, seeds.index(s), [archs.index(a) for a in C.ANCHORS])
                repro.append({"check": "4-anchor cross-arch W seed %d" % s, "published": pub, "recomputed": w,
                              "pass": abs(round(w, 3) - pub) < 1e-9})

        # ---- 2. rater-count-matched W ----
        ca_by_seed = {}
        for si, s in enumerate(seeds):
            vals = []
            for sub in C.all_subsets(range(len(archs)), 5):
                w, _ = cross_arch_w(cube, si, sub)
                if np.isnan(w):
                    continue
                vals.append(w)
                subsets_rows.append({"domain": domain, "seed": s, "subset": "|".join(archs[i] for i in sub), "w": w})
            ca_by_seed[s] = np.array(vals)
        pooled = np.concatenate([v for v in ca_by_seed.values() if len(v)]) if ca_by_seed else np.array([])
        for ai, a in enumerate(archs):
            w, r = cross_seed_w(cube, ai)
            cs_rows.append({"domain": domain, "arch": a, "k_seeds": len(seeds), "w_cross_seed": w})
            if np.isnan(w) or pooled.size == 0:
                continue
            rec = {"domain": domain, "arch": a, "w_cross_seed": w,
                   "pct_in_crossarch_k5_pooled": float(np.mean(pooled < w) + 0.5 * np.mean(pooled == w)),
                   "crossarch_k5_pooled_median": float(np.median(pooled)),
                   "crossarch_k5_pooled_q05": float(np.quantile(pooled, 0.05)),
                   "crossarch_k5_pooled_q95": float(np.quantile(pooled, 0.95)),
                   "n_crossarch_k5": int(pooled.size)}
            rec["above_q95"] = bool(w > rec["crossarch_k5_pooled_q95"])
            rec["within_q05_q95"] = bool(rec["crossarch_k5_pooled_q05"] <= w <= rec["crossarch_k5_pooled_q95"])
            summ_rows.append(rec)

        # ---- 3. permutation null + Friedman for each main W ----
        def add_perm(label, ranks, kind, unit):
            w = C.kendall_w(ranks)
            null = C.w_perm_null(ranks, args.n_perm, rng)
            k, n = ranks.shape
            chi2, pf = C.friedman_from_w(w, k, n)
            pp = C.perm_p(w, null)
            perm_rows.append({"domain": domain, "kind": kind, "unit": unit, "label": label, "k": k, "n": n, "w": w,
                              "null_mean": float(null.mean()), "null_sd": float(null.std()),
                              "w_chance_corrected": (w - null.mean()) / (1 - null.mean()),
                              "p_perm": pp, "friedman_chi2": chi2, "p_friedman": pf})
            pfam.append({"family": "W_vs_chance", "domain": domain, "test": "%s %s" % (kind, label),
                         "p": pp, "source": "w_from_csv perm"})

        for si, s in enumerate(seeds):
            w, r = cross_arch_w(cube, si, range(len(archs)))
            if r is not None:
                add_perm("seed%d" % s, r, "cross_arch_k8", "seed")
        for ai, a in enumerate(archs):
            w, r = cross_seed_w(cube, ai)
            if r is not None:
                add_perm(a, r, "cross_seed_k5", "arch")

        # ---- 4/5. arch-label permutation + nested ANOVA (complete archs only) ----
        for setname, aset in (("anchors4", list(C.ANCHORS)), ("all8", archs)):
            idx = [archs.index(a) for a in aset]
            if not avail[idx].all():
                continue
            y = cube[idx]  # (A,S,M)
            A, S = y.shape[0], y.shape[1]
            rk = np.stack([C.ranks_rows(y[i]) for i in range(A)])  # (A,S,M)
            ties = C.tie_terms(rk)  # (A,S)

            def stat_w(units_rt):
                rr, tt = units_rt
                return float(np.mean(C.kendall_w_batch(rr, tt)))

            units_r = rk.reshape(A * S, M)
            units_t = ties.reshape(A * S)
            obs = float(np.mean(C.kendall_w_batch(rk, ties)))
            null = np.empty(args.n_perm)
            for t in range(args.n_perm):
                perm = rng.permutation(A * S)
                null[t] = np.mean(C.kendall_w_batch(units_r[perm].reshape(A, S, M), units_t[perm].reshape(A, S)))
            p_w = C.perm_p(obs, null)
            alp_rows.append({"domain": domain, "set": setname, "stat": "mean within-arch cross-seed W",
                             "observed": obs, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
                             "excess": obs - float(null.mean()), "p_perm": p_w, "n_perm": args.n_perm})
            pfam.append({"family": "arch_vs_seed", "domain": domain, "test": "arch-label perm W %s" % setname,
                         "p": p_w, "source": "w_from_csv"})

            for yname, yy in (("auroc", y), ("rank", rk)):
                res = nested_anova(yy)
                res.update(crossed_anova_fracs(yy))
                # permutation p for F (shuffle run->arch labels)
                units = yy.reshape(A * S, M)
                fobs = res["F_methodxarch_vs_methodxseed"]
                fnull = np.empty(min(args.n_perm, 5000))
                for t in range(len(fnull)):
                    perm = rng.permutation(A * S)
                    fnull[t] = nested_anova(units[perm].reshape(A, S, M))["F_methodxarch_vs_methodxseed"]
                res["p_F_perm"] = C.perm_p(fobs, fnull)
                res.update({"domain": domain, "set": setname, "y": yname, "A": A, "S": S, "M": M})
                anova_rows.append(res)
                pfam.append({"family": "arch_vs_seed", "domain": domain,
                             "test": "nested F perm %s %s" % (yname, setname), "p": res["p_F_perm"], "source": "w_from_csv"})

        # ---- 6. feature-space rank-1 per cell ----
        for ai, a in enumerate(archs):
            for si, s in enumerate(seeds):
                v = cube[ai, si]
                if np.isnan(v).any():
                    continue
                r = C.ranks_high_is_1(v)
                top = [C.METHODS_ORDER[j] for j in range(M) if r[j] == r.min()]
                fr = {m: r[C.METHODS_ORDER.index(m)] for m in C.FEATURE_METHODS}
                rank1_rows.append({"domain": domain, "arch": a, "seed": s, "top": "|".join(top),
                                   "feature_rank1": bool(min(fr.values()) == 1.0),
                                   "both_features_top2": bool(max(fr.values()) <= 2.0),
                                   "rank_maha": fr["Mahalanobis"], "rank_knn": fr["kNN"],
                                   "gap_best_feature_minus_best_other": float(
                                       max(v[C.METHODS_ORDER.index(m)] for m in C.FEATURE_METHODS)
                                       - max(v[j] for j in range(M) if C.METHODS_ORDER[j] not in C.FEATURE_METHODS))})

        # ---- 7. FPR95-based W (lower FPR95 = better -> rank on -FPR95) ----
        fcube = C.load_cube(root, domain, archs, seeds, "FPR95")
        for si, s in enumerate(seeds):
            ok = [i for i in range(len(archs)) if not np.isnan(fcube[i, si]).any()]
            wa = C.kendall_w(C.ranks_rows(-fcube[ok, si])) if len(ok) >= 2 else np.nan
            wauc = C.kendall_w(C.ranks_rows(cube[ok, si])) if len(ok) >= 2 and not np.isnan(cube[ok, si]).any() else np.nan
            fpr_rows.append({"domain": domain, "kind": "cross_arch", "label": "seed%d" % s, "k": len(ok),
                             "archs_used": "|".join(archs[i] for i in ok), "w_fpr95": wa, "w_auroc_same_raters": wauc})
        for ai, a in enumerate(archs):
            ok = [j for j in range(len(seeds)) if not np.isnan(fcube[ai, j]).any()]
            wa = C.kendall_w(C.ranks_rows(-fcube[ai, ok])) if len(ok) >= 2 else np.nan
            wauc = C.kendall_w(C.ranks_rows(cube[ai, ok])) if len(ok) >= 2 and not np.isnan(cube[ai, ok]).any() else np.nan
            fpr_rows.append({"domain": domain, "kind": "cross_seed", "label": a, "k": len(ok),
                             "archs_used": "|".join(str(seeds[j]) for j in ok), "w_fpr95": wa, "w_auroc_same_raters": wauc})

        # ---- 8. MSP jump tests (Camelyon definition: MSP - that seed's ResNet mean) ----
        mi = C.METHODS_ORDER.index("MSP")
        rmean = np.nanmean(cube[[archs.index(a) for a in C.RESNETS], :, mi], axis=0)
        for ai, a in enumerate(archs):
            if a in C.RESNETS:
                continue
            j = cube[ai, :, mi] - rmean
            j = j[~np.isnan(j)]
            if len(j) < 2:
                continue
            sd = float(np.std(j, ddof=1))
            se = sd / np.sqrt(len(j))
            try:
                from scipy.stats import t as tdist

                p0 = float(tdist.sf(np.mean(j) / se, len(j) - 1)) if se > 0 else float("nan")
                p15 = float(tdist.sf((np.mean(j) - C.JUMP_DELTA) / se, len(j) - 1)) if se > 0 else float("nan")
            except Exception:
                p0 = p15 = float("nan")
            # sensitivity (reviewer: the per-seed ResNet baseline is itself noisy): compare this arch's
            # 5 MSP values with the pooled 10 ResNet cells (Welch, one-sided). Descriptive only, no family.
            x = cube[ai, :, mi][~np.isnan(cube[ai, :, mi])]
            rpool = cube[[archs.index(r) for r in C.RESNETS], :, mi].ravel()
            rpool = rpool[~np.isnan(rpool)]
            try:
                from scipy.stats import ttest_ind

                tw = ttest_ind(x, rpool, equal_var=False)
                pw = float(tw.pvalue / 2.0) if tw.statistic > 0 else float(1.0 - tw.pvalue / 2.0)
            except Exception:
                pw = float("nan")
            jump_rows.append({"domain": domain, "arch": a, "n_seeds": len(j), "jump_mean": float(np.mean(j)),
                              "jump_sd": sd, "n_ge_0.15": int(np.sum(j >= C.JUMP_DELTA)),
                              "p_jump_gt0": p0, "p_jump_gt0.15": p15,
                              "jump_vs_pooled_resnet": float(np.mean(x) - np.mean(rpool)) if len(rpool) else float("nan"),
                              "p_welch_vs_pooled_resnet_sens": pw})
            pfam.append({"family": "msp_jump_gt0", "domain": domain, "test": "jump>0 %s" % a, "p": p0, "source": "w_from_csv t"})
            pfam.append({"family": "msp_jump_gt0.15", "domain": domain, "test": "jump>0.15 %s" % a, "p": p15, "source": "w_from_csv t"})

    # ---- write ----
    tabs = {"repro_checks.csv": repro, "matched_w_subsets.csv": subsets_rows, "cross_seed_w_all.csv": cs_rows,
            "matched_w_summary.csv": summ_rows, "w_perm_friedman.csv": perm_rows, "arch_label_perm.csv": alp_rows,
            "variance_decomposition.csv": anova_rows, "feature_rank1_cells.csv": rank1_rows,
            "w_fpr95.csv": fpr_rows, "msp_jump_tests.csv": jump_rows, "pvalues_family.csv": pfam}
    for name, rows in tabs.items():
        pd.DataFrame(rows).to_csv(out / name, index=False)

    rp = pd.DataFrame(repro)
    log.append("")
    log.append("REPRODUCTION CHECKS (must all PASS before reading anything else):")
    for _, r in rp.iterrows():
        log.append("  REPRO %-4s %-40s published %.3f recomputed %.4f" % ("PASS" if r["pass"] else "FAIL", r["check"],
                                                                      r["published"], r["recomputed"]))
    log.append("")
    log.append("Wrote: " + ", ".join(sorted(tabs)))
    log.append("Read results only against decisions/decision_precommit_rigor_pack.md.")
    C.write_text(out / "w_from_csv_log.txt", log)
    print("\n".join(log))
    if len(rp) and not rp["pass"].all():
        sys.exit(2)


if __name__ == "__main__":
    main()
