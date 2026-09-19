#!/usr/bin/env python3
"""Rebuild architecture-invariance tables.

Primary metric is Kendall's W (rank concordance across backbones in one
domain), not mean AUROC-spread pooled over domains. Spread stays as a
per-domain supplement. Do not treat a 3-domain mean spread as a method
property — dropping Camelyon reverses Maha vs MSP.

Does not touch CIFAR-10-C / BN / ECE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.cnn_family import ARCH_FAMILY, ARCH_FAMILY_LABEL

ROOT = Path("outputs/reports")
LONG_OUT = ROOT / "architecture_invariance_long.csv"
WIDE_OUT = ROOT / "architecture_invariance.csv"
RANK_OUT = ROOT / "architecture_invariance_ranks.csv"
W_OUT = ROOT / "architecture_invariance_kendall_w.csv"
LODO_OUT = ROOT / "architecture_invariance_leave_one_domain.csv"
FAMILY_OUT = ROOT / "architecture_invariance_family_msp.csv"
SCENARIO_OUT = ROOT / "architecture_invariance_scenario.txt"
N_BOOT = 1000
SEED = 42

DISPLAY = {
    "msp": "MSP",
    "energy": "Energy",
    "logit_norm": "ELogitNorm",
    "vim": "ViM",
    "react_energy": "ReAct",
    "mahalanobis": "Mahalanobis",
    "knn": "kNN",
}
INV_DISPLAY = {v: k for k, v in DISPLAY.items()}
METHODS_ORDER = list(DISPLAY.values())
LOGIT_METHODS = ("MSP", "Energy", "ELogitNorm")
FEATURE_METHODS = ("Mahalanobis", "kNN")
JUMP_DELTA = 0.15
FEATURE_DRAG = 0.10

SCORE_GLOBS = (
    (ROOT / "resnet18_score_comparison.csv", "skin_isic_pad", "resnet18"),
    (ROOT / "resnet50_score_comparison.csv", "skin_isic_pad", "resnet50"),
    (ROOT / "effb3_score_comparison.csv", "skin_isic_pad", "effb3"),
)


def _official_backbone(stem: str) -> str | None:
    """Map a score-CSV stem to the official zoo name, or drop it.

    Official Camelyon/skin W is 8 CNNs. EffV2-S uses the @224 cell
    (native 384 is the resolution trap). EffB3@224 and ViT-B/16 and
    SupCon are held-out evidence, not extra W columns. iWildCam is a
    separate bound and is not scanned here.

    Rounding trap: official R50 MSP 0.514974
    (camelyon17/frac1/seed42/resnet50_score_comparison.csv) and Job-B
    R18-SupCon MSP 0.515415 (resnet18_supcon_score_comparison.csv) both
    print as 0.515 at 3 d.p. Never substitute the SupCon file for R50.
    """
    if stem.endswith("_supcon") or stem == "vit_b_16":
        return None
    if "openmibood" in stem or "native50" in stem:
        return None
    if stem == "effb3_224":
        return None
    if stem == "efficientnet_v2_s":
        return None
    if stem == "efficientnet_v2_s_224":
        return "efficientnet_v2_s"
    if stem.rsplit("_", 1)[-1].isdigit():
        return None
    return stem


def _scan() -> list[tuple[Path, str, str]]:
    found = list(SCORE_GLOBS)
    for path in (ROOT / "skin").glob("*_score_comparison.csv"):
        bb = _official_backbone(path.name.replace("_score_comparison.csv", ""))
        if bb is None:
            continue
        found.append((path, "skin_isic_pad", bb))
    for path in (ROOT / "camelyon17" / "frac1" / "seed42").glob("*_score_comparison.csv"):
        bb = _official_backbone(path.name.replace("_score_comparison.csv", ""))
        if bb is None:
            continue
        found.append((path, "camelyon17", bb))
    for path in (ROOT / "midog" / "seed42").glob("*_score_comparison.csv"):
        bb = _official_backbone(path.name.replace("_score_comparison.csv", ""))
        if bb is None:
            continue
        found.append((path, "midog", bb))
    return found


def _load_wide_seed(records: dict) -> None:
    path = WIDE_OUT
    if not path.exists():
        return
    df = pd.read_csv(path)
    colmap = {
        "auroc_r18": "resnet18",
        "auroc_r50": "resnet50",
        "auroc_effb3": "effb3",
    }
    for _, row in df.iterrows():
        method = INV_DISPLAY.get(str(row["method"]), str(row["method"]))
        if method not in DISPLAY:
            continue
        domain = str(row["domain"])
        for col, bb in colmap.items():
            if col in row.index and pd.notna(row[col]):
                records[(domain, method, bb)] = float(row[col])
        for col in row.index:
            if not str(col).startswith("auroc_"):
                continue
            bb = str(col)[len("auroc_") :]
            if bb in {"r18", "r50", "effb3"}:
                continue
            if pd.notna(row[col]):
                records[(domain, method, bb)] = float(row[col])


def _bootstrap_std(vals: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    point = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if len(vals) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(vals)
    for i in range(n_boot):
        samp = vals[rng.integers(0, n, n)]
        boots[i] = float(np.std(samp, ddof=1)) if n > 1 else 0.0
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _ranks_high_is_1(values: np.ndarray) -> np.ndarray:
    """Average ranks; 1 = highest AUROC."""
    n = len(values)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + 1 + j + 1)
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Tie-corrected Kendall's W. rank_matrix: (n_raters/backbones, n_items/methods)."""
    k, n = rank_matrix.shape
    if k < 2 or n < 2:
        return float("nan")
    r = rank_matrix.astype(np.float64)
    s = float(np.sum((r.sum(axis=0) - r.sum() / n) ** 2))
    tie_term = 0.0
    for j in range(k):
        _, counts = np.unique(r[j], return_counts=True)
        tie_term += float(np.sum(counts**3 - counts))
    denom = k**2 * (n**3 - n) - k * tie_term
    if denom <= 0:
        return float("nan")
    return float(12.0 * s / denom)


def _scenario_camelyon(long_df: pd.DataFrame) -> tuple[str, str]:
    cam = long_df[long_df["domain"] == "camelyon17"]
    bbs = sorted(cam["backbone"].unique())
    n = len(bbs)
    if n < 8:
        return (
            "pending_n8",
            f"Camelyon has {n} backbone(s); A/B/C waits for n=8. "
            "Skin n=8 is in: broad claim (Maha rank-stable) confirmed; "
            "narrow MSP-jump stays a Camelyon case study, not a cross-domain law.",
        )
    msp = cam[cam["method"] == "MSP"].set_index("backbone")["auroc"]
    need = ("resnet18", "resnet50")
    if any(b not in msp.index for b in need):
        return "pending_resnet", "Missing ResNet-like MSP cells."
    resnet_mean = float(msp.loc[list(need)].mean())
    jumped = [bb for bb in msp.index if float(msp.loc[bb]) - resnet_mean >= JUMP_DELTA]
    eff_family = {"effb3", "efficientnet_v2_s"}
    other = set(msp.index) - set(need) - eff_family
    jumped_eff = [bb for bb in jumped if bb in eff_family]
    jumped_other = [bb for bb in jumped if bb in other]

    orig = [b for b in ("resnet18", "resnet50", "effb3") if b in msp.index]
    dragged = []
    for method in FEATURE_METHODS:
        sub = cam[cam["method"] == method].set_index("backbone")["auroc"]
        if not orig or any(b not in sub.index for b in orig):
            continue
        med = float(sub.loc[orig].median())
        for bb in sub.index:
            if bb in orig:
                continue
            if float(sub.loc[bb]) < med - FEATURE_DRAG:
                dragged.append(f"{method}/{bb}")

    if dragged:
        return (
            "C",
            "Feature-based AUROC dragged by a new backbone "
            f"({', '.join(dragged)}). Maha>MSP is not a fallback "
            "(OpenMIBOOD §5). Rescue with Direction #1/#3 or there is no paper. "
            "See decision_precommit_camelyon8_skin8.md.",
        )
    if jumped_other:
        return (
            "A",
            "MSP-style jump appears in >=1 family besides EfficientNet "
            f"(jumped={jumped}); check that feature-based ranks stay flat.",
        )
    if jumped_eff and not jumped_other:
        return (
            "B",
            "Jump confined to EfficientNet-like family "
            f"(jumped={jumped_eff}). Architecture-specific anomaly, not invariance.",
        )
    return (
        "no_jump",
        f"No MSP jump ≥{JUMP_DELTA} vs ResNet-like mean ({resnet_mean:.3f}). "
        "EffB3 cell may be isolated; still not a cross-domain law.",
    )


def main() -> None:
    records: dict[tuple[str, str, str], float] = {}
    _load_wide_seed(records)
    for path, domain, backbone in _scan():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "Method" not in df.columns or "AUROC" not in df.columns:
            continue
        for _, row in df.iterrows():
            method = str(row["Method"])
            if method not in DISPLAY:
                continue
            records[(domain, method, backbone)] = float(row["AUROC"])

    if not records:
        raise RuntimeError("No AUROC records found")

    long_rows = []
    for (d, m, b), a in sorted(records.items()):
        long_rows.append(
            {
                "domain": d,
                "method": DISPLAY[m],
                "backbone": b,
                "family": ARCH_FAMILY.get(b, "unknown"),
                "family_label": ARCH_FAMILY_LABEL.get(
                    ARCH_FAMILY.get(b, "unknown"), ARCH_FAMILY.get(b, "unknown")
                ),
                "auroc": a,
            }
        )
    long_df = pd.DataFrame(long_rows)
    LONG_OUT.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(LONG_OUT, index=False)

    summary_rows = []
    rank_rows = []
    w_rows = []
    for (domain, method), sub in long_df.groupby(["domain", "method"], sort=False):
        vals = sub["auroc"].to_numpy()
        spread, lo, hi = _bootstrap_std(vals)
        rec = {
            "method": method,
            "domain": domain,
            "n_backbones": int(len(vals)),
            "mean_auroc": float(np.mean(vals)),
            "spread": spread,
            "spread_ci_lo": lo,
            "spread_ci_hi": hi,
        }
        for _, r in sub.iterrows():
            rec[f"auroc_{r['backbone']}"] = float(r["auroc"])
        summary_rows.append(rec)

    for domain, gdom in long_df.groupby("domain", sort=False):
        methods = [m for m in METHODS_ORDER if m in set(gdom["method"])]
        backbones = sorted(gdom["backbone"].unique())
        mat = np.full((len(backbones), len(methods)), np.nan)
        for i, bb in enumerate(backbones):
            for j, method in enumerate(methods):
                hit = gdom[(gdom["backbone"] == bb) & (gdom["method"] == method)]
                if hit.empty:
                    continue
                mat[i, j] = float(hit["auroc"].iloc[0])
        complete = ~np.isnan(mat).any(axis=1)
        mat_c = mat[complete]
        bbs_c = [b for b, ok in zip(backbones, complete) if ok]
        if len(mat_c) >= 1:
            for i, bb in enumerate(bbs_c):
                ranks = _ranks_high_is_1(mat_c[i])
                for j, method in enumerate(methods):
                    rank_rows.append(
                        {
                            "domain": domain,
                            "backbone": bb,
                            "family": ARCH_FAMILY.get(bb, "unknown"),
                            "method": method,
                            "auroc": float(mat_c[i, j]),
                            "rank": float(ranks[j]),
                        }
                    )
        w = kendall_w(_ranks_high_is_1_rows(mat_c)) if len(mat_c) >= 2 else float("nan")
        w_rows.append(
            {
                "domain": domain,
                "n_backbones": int(len(mat_c)),
                "n_methods": int(len(methods)),
                "kendall_w": w,
                "primary_metric": "kendall_w",
                "note": (
                    "W=1 identical method ranking across backbones; "
                    "W=0 no concordance. Do not pool W across domains."
                ),
            }
        )

    wide = pd.DataFrame(summary_rows)
    wide.to_csv(WIDE_OUT, index=False)
    ranks_df = pd.DataFrame(rank_rows)
    ranks_df.to_csv(RANK_OUT, index=False)
    w_df = pd.DataFrame(w_rows)
    w_df.to_csv(W_OUT, index=False)

    lodo_rows = []
    domains = sorted(long_df["domain"].unique())
    for method, gm in long_df.groupby("method", sort=False):
        per = {}
        for domain, gd in gm.groupby("domain"):
            if len(gd) >= 2:
                per[domain] = float(np.std(gd["auroc"].to_numpy(), ddof=1))
        if not per:
            continue
        mean_all = float(np.mean(list(per.values())))
        lodo_rows.append(
            {
                "method": method,
                "held_out": "(none — mean of per-domain spreads)",
                "mean_spread": mean_all,
                "n_domains": len(per),
                "warning": "NOT a method property; Camelyon-driven at n=3",
            }
        )
        for hold in domains:
            kept = [per[d] for d in per if d != hold]
            if not kept:
                continue
            lodo_rows.append(
                {
                    "method": method,
                    "held_out": hold,
                    "mean_spread": float(np.mean(kept)),
                    "n_domains": len(kept),
                    "warning": "leave-one-domain diagnostic only",
                }
            )
    pd.DataFrame(lodo_rows).to_csv(LODO_OUT, index=False)

    fam_rows = []
    msp = long_df[long_df["method"] == "MSP"]
    for domain, gd in msp.groupby("domain"):
        for _, r in gd.iterrows():
            fam_rows.append(
                {
                    "domain": domain,
                    "backbone": r["backbone"],
                    "family": r["family"],
                    "family_label": r["family_label"],
                    "msp_auroc": r["auroc"],
                }
            )
    pd.DataFrame(fam_rows).to_csv(FAMILY_OUT, index=False)

    code, msg = _scenario_camelyon(long_df)
    skin_n = int(long_df[long_df["domain"] == "skin_isic_pad"]["backbone"].nunique())
    cam_n = int(long_df[long_df["domain"] == "camelyon17"]["backbone"].nunique())
    header = (
        f"scenario={code}\n"
        f"camelyon_n_backbones={cam_n}\n"
        f"skin_n_backbones={skin_n}\n"
        f"{msg}\n"
        "Official W uses EffV2-S@224 on Camelyon and skin; native-384 dropped. "
        "EffB3 Camelyon official cell remains native 300; EffB3@224 is appendix. "
        "ViT-B/16 and SupCon are not W columns.\n"
        "Phao A CLOSED: FM GigaPath/Phikon Camelyon MSP 0.664/0.605 both >0.5; "
        "MIDOG-only footnote; no invert debate.\n"
        "See decision_precommit_camelyon8_skin8.md (A/B/C + floor revoked). "
        "See outputs/reports/NARRATIVE_LOCKS.txt (rebuild must not overwrite).\n"
    )
    SCENARIO_OUT.write_text(header)

    print(f"Wrote {LONG_OUT} ({len(long_df)} rows)")
    print(f"Wrote {WIDE_OUT} ({len(wide)} rows)")
    print(f"Wrote {RANK_OUT} ({len(ranks_df)} rows)")
    print(f"Wrote {W_OUT}")
    print(f"Wrote {LODO_OUT}")
    print(f"Wrote {FAMILY_OUT}")
    print(f"Wrote {SCENARIO_OUT}")
    print(header)

    print("Kendall's W (primary, per domain — not pooled):")
    print(w_df.to_string(index=False))
    print("\nPer-domain spread (supplement only):")
    for domain, g in wide.groupby("domain", sort=False):
        g = g.sort_values("spread")
        print(f"\n=== {domain} (n_backbones max={int(g['n_backbones'].max())}) ===")
        cols = ["method", "n_backbones", "mean_auroc", "spread"]
        print(g[cols].to_string(index=False))
        if not ranks_df.empty:
            pivot = ranks_df[ranks_df["domain"] == domain].pivot(
                index="method", columns="backbone", values="rank"
            )
            print("ranks (1=best):")
            print(pivot.reindex(METHODS_ORDER).to_string())


def _ranks_high_is_1_rows(mat: np.ndarray) -> np.ndarray:
    out = np.empty_like(mat, dtype=np.float64)
    for i in range(len(mat)):
        out[i] = _ranks_high_is_1(mat[i])
    return out


if __name__ == "__main__":
    main()
