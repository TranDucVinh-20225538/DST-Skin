#!/usr/bin/env python3
"""Permutation null for official Kendall W. CPU, no retrain, official W unchanged.

Two nulls per domain, n_perm=10000, seed=42:
  unconstrained: independently permute method ranks within each backbone.
  feature_locked: keep Maha/kNN ranks as observed; shuffle the other five
    among their observed rank slots.

Also reports W on logit-only and feature-only method subsets (no permutation).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "rebuild_architecture_invariance", HERE / "rebuild_architecture_invariance.py"
)
_REBUILD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_REBUILD)

METHODS_ORDER = _REBUILD.METHODS_ORDER
FEATURE = _REBUILD.FEATURE_METHODS
LOGIT = _REBUILD.LOGIT_METHODS
_ranks_high_is_1_rows = _REBUILD._ranks_high_is_1_rows
kendall_w = _REBUILD.kendall_w

RANKS = Path("outputs/reports/architecture_invariance_ranks.csv")
OUT_TXT = Path("outputs/reports/kendall_w_perm.txt")
OUT_CSV = Path("outputs/reports/kendall_w_perm.csv")
N_PERM = 10_000
SEED = 42
DOMAINS = ("camelyon17", "midog", "skin_isic_pad")


def load_matrix(df: pd.DataFrame, domain: str) -> tuple[np.ndarray, list[str], list[str]]:
    sub = df[df["domain"] == domain]
    methods = [m for m in METHODS_ORDER if m in set(sub["method"])]
    backbones = sorted(sub["backbone"].unique())
    mat = np.full((len(backbones), len(methods)), np.nan)
    for i, bb in enumerate(backbones):
        for j, method in enumerate(methods):
            hit = sub[(sub["backbone"] == bb) & (sub["method"] == method)]
            mat[i, j] = float(hit["auroc"].iloc[0])
    if np.isnan(mat).any():
        raise RuntimeError(f"Incomplete rank matrix: {domain}")
    return mat, backbones, methods


def perm_p(observed: float, null: np.ndarray, alternative: str) -> float:
    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    else:
        hits = int(np.sum(null <= observed))
    return float((hits + 1) / (len(null) + 1))


def run_domain(df: pd.DataFrame, domain: str) -> dict:
    mat, backbones, methods = load_matrix(df, domain)
    ranks = _ranks_high_is_1_rows(mat)
    w_obs = kendall_w(ranks)
    idx = {m: j for j, m in enumerate(methods)}
    feat_j = [idx[m] for m in FEATURE if m in idx]
    logit_j = [idx[m] for m in LOGIT if m in idx]
    other_j = [j for j, m in enumerate(methods) if m not in FEATURE]

    w_feat = (
        kendall_w(_ranks_high_is_1_rows(mat[:, feat_j])) if len(feat_j) >= 2 else float("nan")
    )
    w_logit = (
        kendall_w(_ranks_high_is_1_rows(mat[:, logit_j])) if len(logit_j) >= 2 else float("nan")
    )

    rng = np.random.default_rng(SEED)
    k, n = ranks.shape
    null_free = np.empty(N_PERM)
    null_lock = np.empty(N_PERM)
    for t in range(N_PERM):
        shuf = np.empty_like(ranks)
        for i in range(k):
            shuf[i] = rng.permutation(n) + 1.0
        null_free[t] = kendall_w(shuf)

        locked = ranks.copy()
        for i in range(k):
            locked[i, other_j] = rng.permutation(ranks[i, other_j])
        null_lock[t] = kendall_w(locked)

    chi2 = float(k * (n - 1) * w_obs)
    return {
        "domain": domain,
        "n_backbones": k,
        "n_methods": n,
        "w_obs": w_obs,
        "w_feature_only": w_feat,
        "w_logit_only": w_logit,
        "friedman_chi2": chi2,
        "null_free_mean": float(null_free.mean()),
        "null_free_sd": float(null_free.std()),
        "p_free_greater": perm_p(w_obs, null_free, "greater"),
        "null_lock_mean": float(null_lock.mean()),
        "null_lock_sd": float(null_lock.std()),
        "p_lock_greater": perm_p(w_obs, null_lock, "greater"),
        "p_lock_less": perm_p(w_obs, null_lock, "less"),
        "n_perm": N_PERM,
        "backbones": ",".join(backbones),
        "methods": ",".join(methods),
    }


def main() -> None:
    df = pd.read_csv(RANKS)
    rows = [run_domain(df, domain) for domain in DOMAINS if domain in set(df["domain"])]
    lines = [
        "Official-seed Kendall W permutation null. Official W files unchanged.",
        f"n_perm={N_PERM} seed={SEED}",
        "unconstrained: random ranks per backbone. p(W>=obs)<<0.05 = concordance exists.",
        "feature_locked: Maha/kNN ranks fixed; other 5 shuffled in their slots.",
        "If W_obs sits in the locked null, most of W is the Maha/kNN lock.",
        "",
    ]
    for r in rows:
        lines.extend(
            [
                f"=== {r['domain']} n={r['n_backbones']} methods={r['n_methods']} ===",
                f"W_obs={r['w_obs']:.6f}",
                f"W_feature_only (Maha,kNN)={r['w_feature_only']:.6f}",
                f"W_logit_only (MSP,Energy,ELogitNorm)={r['w_logit_only']:.6f}",
                f"Friedman chi2=k(n-1)W={r['friedman_chi2']:.3f} df={r['n_methods']-1}",
                f"unconstrained null: mean={r['null_free_mean']:.4f} sd={r['null_free_sd']:.4f} "
                f"p(W>=obs)={r['p_free_greater']:.4g}",
                f"feature_locked null: mean={r['null_lock_mean']:.4f} sd={r['null_lock_sd']:.4f} "
                f"p(W>=obs)={r['p_lock_greater']:.4g} p(W<=obs)={r['p_lock_less']:.4g}",
                f"backbones={r['backbones']}",
                "",
            ]
        )
    text = "\n".join(lines)
    OUT_TXT.write_text(text)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(text)
    print(f"Wrote {OUT_TXT} {OUT_CSV}")


if __name__ == "__main__":
    main()
