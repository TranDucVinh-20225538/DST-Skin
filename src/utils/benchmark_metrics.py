"""Shared OOD benchmark metrics: bootstrap CI and dispersion/separability.

LOCKED 2026-09-17 — do not retune without a new ablation:
  Primary dispersion metric: LogitGap, 1% tail-trimmed mean (same formula as raw).
  CVID is appendix-only after estimator-sensitivity (raw mean/std, trim-1%, MAD/median).
  MAD/median is not a drop-in CVID replacement (sign-flip on clean skin R18).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, roc_auc_score, silhouette_score

BOOTSTRAP_METHODS = ("msp", "energy", "mahalanobis", "knn")
N_BOOTSTRAP = 1000
N_SILHOUETTE_PERM = 1000
SILHOUETTE_MAX_PER_CLASS = 2500
# Locked: 1% each tail. LogitGap = primary. CVID = appendix (trimmed mean/std, not MAD).
ROBUST_TRIM_FRAC = 0.01


def calc_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    id_scores = np.asarray(id_scores, dtype=np.float32)
    ood_scores = np.asarray(ood_scores, dtype=np.float32)
    y_true = np.concatenate([
        np.ones(len(id_scores), dtype=np.int32),
        np.zeros(len(ood_scores), dtype=np.int32),
    ])
    y_score = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_score))


def calc_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    id_scores = np.asarray(id_scores, dtype=np.float32)
    ood_scores = np.asarray(ood_scores, dtype=np.float32)
    thr = float(np.percentile(id_scores, 5))
    fpr = float(np.mean(ood_scores > thr))
    return fpr, thr


def bootstrap_dispersion_gap_ci(
    id_logits: np.ndarray,
    ood_logits: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
    ci: float = 95.0,
) -> dict[str, float]:
    """Percentile bootstrap CI for CVID and LogitGap gaps (OOD minus ID)."""
    id_logits = np.asarray(id_logits, dtype=np.float64)
    ood_logits = np.asarray(ood_logits, dtype=np.float64)
    n_id, n_ood = len(id_logits), len(ood_logits)
    rng = np.random.default_rng(seed)

    id_cvid = compute_cvid_logit_norm(per_sample_logit_norm(id_logits))
    ood_cvid = compute_cvid_logit_norm(per_sample_logit_norm(ood_logits))
    id_gap_mean = aggregate_logit_gap(per_sample_logit_gap(id_logits))
    ood_gap_mean = aggregate_logit_gap(per_sample_logit_gap(ood_logits))

    cvid_gaps = np.empty(n_boot, dtype=np.float64)
    logit_gap_gaps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        id_b = id_logits[rng.integers(0, n_id, n_id)]
        ood_b = ood_logits[rng.integers(0, n_ood, n_ood)]
        cvid_gaps[i] = (
            compute_cvid_logit_norm(per_sample_logit_norm(ood_b))
            - compute_cvid_logit_norm(per_sample_logit_norm(id_b))
        )
        logit_gap_gaps[i] = (
            aggregate_logit_gap(per_sample_logit_gap(ood_b))
            - aggregate_logit_gap(per_sample_logit_gap(id_b))
        )

    alpha = (100.0 - ci) / 2.0
    return {
        "cvid_gap_vs_id": float(ood_cvid - id_cvid),
        "cvid_gap_vs_id_CI_low": float(np.percentile(cvid_gaps, alpha)),
        "cvid_gap_vs_id_CI_high": float(np.percentile(cvid_gaps, 100.0 - alpha)),
        "logit_gap_mean_gap_vs_id": float(ood_gap_mean - id_gap_mean),
        "logit_gap_mean_gap_vs_id_CI_low": float(np.percentile(logit_gap_gaps, alpha)),
        "logit_gap_mean_gap_vs_id_CI_high": float(
            np.percentile(logit_gap_gaps, 100.0 - alpha)
        ),
    }


def bootstrap_ci(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
    ci: float = 95.0,
) -> dict[str, float]:
    """Percentile bootstrap on resampled ID/OOD score vectors."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_scores), len(ood_scores)
    rng = np.random.default_rng(seed)

    aurocs = np.empty(n_boot, dtype=np.float64)
    fprs = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        id_idx = rng.integers(0, n_id, n_id)
        ood_idx = rng.integers(0, n_ood, n_ood)
        id_b = id_scores[id_idx]
        ood_b = ood_scores[ood_idx]
        aurocs[i] = calc_auroc(id_b, ood_b)
        fprs[i], _ = calc_fpr95(id_b, ood_b)

    alpha = (100.0 - ci) / 2.0
    return {
        "AUROC": float(calc_auroc(id_scores, ood_scores)),
        "AUROC_CI_low": float(np.percentile(aurocs, alpha)),
        "AUROC_CI_high": float(np.percentile(aurocs, 100.0 - alpha)),
        "FPR95": float(calc_fpr95(id_scores, ood_scores)[0]),
        "FPR95_CI_low": float(np.percentile(fprs, alpha)),
        "FPR95_CI_high": float(np.percentile(fprs, 100.0 - alpha)),
    }


def per_sample_logit_norm(logits: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(logits, dtype=np.float64), axis=1)


def trim_tails(x: np.ndarray, proportiontocut: float = ROBUST_TRIM_FRAC) -> np.ndarray:
    """Drop proportiontocut from each tail (by sorted value), shared trim for CVID/LogitGap."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 3:
        return x
    k = int(np.floor(proportiontocut * n))
    if k <= 0 or 2 * k >= n:
        return x
    return x[k : n - k]


def compute_cvid_logit_norm(logit_norm: np.ndarray) -> float:
    """Between-sample CV of L2 logit norms: std/mean on 1%-trimmed norms (not MAD/median)."""
    trimmed = trim_tails(logit_norm, ROBUST_TRIM_FRAC)
    if len(trimmed) == 0:
        return 0.0
    mean = float(np.mean(trimmed))
    std = float(np.std(trimmed))
    return std / (np.abs(mean) + 1e-8)


def per_sample_logit_gap(logits: np.ndarray) -> np.ndarray:
    """Per-sample LogitGap: max_logit - mean(other logits), literature-aligned."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    sorted_logits = np.sort(logits, axis=1)
    top1 = sorted_logits[:, -1]
    rest_mean = sorted_logits[:, :-1].mean(axis=1)
    return top1 - rest_mean


def aggregate_logit_gap(gaps: np.ndarray) -> float:
    """Robust aggregate of per-sample LogitGap: trimmed mean (ROBUST_TRIM_FRAC each tail)."""
    from scipy.stats import trim_mean

    gaps = np.asarray(gaps, dtype=np.float64)
    if len(gaps) == 0:
        return 0.0
    if len(gaps) < 3:
        return float(np.mean(gaps))
    return float(trim_mean(gaps, proportiontocut=ROBUST_TRIM_FRAC))


def compute_logit_gap_stats(logits: np.ndarray) -> dict[str, float]:
    gaps = per_sample_logit_gap(logits)
    trimmed_gaps = trim_tails(gaps, ROBUST_TRIM_FRAC)
    return {
        "logit_gap_mean": aggregate_logit_gap(gaps),
        "logit_gap_std": float(np.std(trimmed_gaps)) if len(trimmed_gaps) else 0.0,
    }


def compute_logit_dispersion(logits: np.ndarray) -> dict[str, float]:
    logit_norm = per_sample_logit_norm(logits)
    return {
        "logit_norm_mean": float(np.mean(logit_norm)),
        "logit_norm_std": float(np.std(logit_norm)),
        "logit_norm_median": float(np.median(logit_norm)),
        "cvid_logit_norm": compute_cvid_logit_norm(logit_norm),
        **compute_logit_gap_stats(logits),
    }


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def subsample_balanced_feats(
    id_feats: np.ndarray,
    ood_feats: np.ndarray,
    max_per_class: int | None = SILHOUETTE_MAX_PER_CLASS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap per-class N for silhouette/permutation (full-set metrics unchanged)."""
    id_feats = np.asarray(id_feats)
    ood_feats = np.asarray(ood_feats)
    if max_per_class is None or max_per_class <= 0:
        return id_feats, ood_feats
    rng = np.random.default_rng(seed)
    if len(id_feats) > max_per_class:
        id_feats = id_feats[rng.choice(len(id_feats), max_per_class, replace=False)]
    if len(ood_feats) > max_per_class:
        ood_feats = ood_feats[rng.choice(len(ood_feats), max_per_class, replace=False)]
    return id_feats, ood_feats


def silhouette_permutation_test(
    id_feats: np.ndarray,
    ood_feats: np.ndarray,
    n_perm: int = N_SILHOUETTE_PERM,
    seed: int = 42,
    max_per_class: int | None = SILHOUETTE_MAX_PER_CLASS,
) -> dict[str, float]:
    """Silhouette ID vs OOD plus label-shuffle null (p = P(null >= observed))."""
    id_sub, ood_sub = subsample_balanced_feats(
        id_feats, ood_feats, max_per_class=max_per_class, seed=seed
    )
    id_n = l2_normalize(id_sub)
    ood_n = l2_normalize(ood_sub)
    feats = np.vstack([id_n, ood_n])
    labels = np.array([0] * len(id_n) + [1] * len(ood_n), dtype=np.int32)
    dist = pairwise_distances(feats, metric="cosine")
    observed = float(silhouette_score(dist, labels, metric="precomputed"))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        null[i] = silhouette_score(dist, rng.permutation(labels), metric="precomputed")

    p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {
        "feature_silhouette_cosine": observed,
        "silhouette_n_id": len(id_n),
        "silhouette_n_ood": len(ood_n),
        "silhouette_perm_p_value": p_value,
        "silhouette_perm_null_mean": float(np.mean(null)),
        "silhouette_perm_null_std": float(np.std(null)),
    }


def build_dispersion_vs_separability_table(
    val_logits: np.ndarray,
    ood_logits: np.ndarray,
    val_feats: np.ndarray,
    ood_feats: np.ndarray,
    n_silhouette_perm: int = N_SILHOUETTE_PERM,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    id_stats = compute_logit_dispersion(val_logits)
    ood_stats = compute_logit_dispersion(ood_logits)
    sep = silhouette_permutation_test(
        val_feats, ood_feats, n_perm=n_silhouette_perm, seed=seed
    )
    gap_ci = bootstrap_dispersion_gap_ci(
        val_logits, ood_logits, n_boot=n_bootstrap, seed=seed
    )

    gap_ci_nan = {
        "cvid_gap_vs_id_CI_low": np.nan,
        "cvid_gap_vs_id_CI_high": np.nan,
        "logit_gap_mean_gap_vs_id_CI_low": np.nan,
        "logit_gap_mean_gap_vs_id_CI_high": np.nan,
    }

    rows = [
        {
            "split": "ID",
            "n_samples": len(val_logits),
            **id_stats,
            **sep,
            "cvid_gap_vs_id": 0.0,
            "logit_norm_mean_gap_vs_id": 0.0,
            "logit_gap_mean_gap_vs_id": 0.0,
            **gap_ci_nan,
        },
        {
            "split": "OOD",
            "n_samples": len(ood_logits),
            **ood_stats,
            **sep,
            "cvid_gap_vs_id": gap_ci["cvid_gap_vs_id"],
            "cvid_gap_vs_id_CI_low": gap_ci["cvid_gap_vs_id_CI_low"],
            "cvid_gap_vs_id_CI_high": gap_ci["cvid_gap_vs_id_CI_high"],
            "logit_norm_mean_gap_vs_id": (
                ood_stats["logit_norm_mean"] - id_stats["logit_norm_mean"]
            ),
            "logit_gap_mean_gap_vs_id": gap_ci["logit_gap_mean_gap_vs_id"],
            "logit_gap_mean_gap_vs_id_CI_low": gap_ci[
                "logit_gap_mean_gap_vs_id_CI_low"
            ],
            "logit_gap_mean_gap_vs_id_CI_high": gap_ci[
                "logit_gap_mean_gap_vs_id_CI_high"
            ],
        },
    ]
    return pd.DataFrame(rows)


def _is_per_sample(scores: np.ndarray) -> bool:
    return np.asarray(scores).ndim > 0 and np.asarray(scores).size > 1


def build_score_comparison_df(
    scores_id: dict,
    scores_ood: dict,
    methods: list[str],
    bootstrap_methods: tuple[str, ...] = BOOTSTRAP_METHODS,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        if method not in scores_id or method not in scores_ood:
            continue
        id_s = np.asarray(scores_id[method], dtype=np.float64)
        ood_s = np.asarray(scores_ood[method], dtype=np.float64)
        if not _is_per_sample(id_s) or not _is_per_sample(ood_s):
            continue

        auroc = calc_auroc(id_s, ood_s)
        fpr95, thr = calc_fpr95(id_s, ood_s)
        row = {
            "Method": method,
            "AUROC": float(auroc),
            "FPR95": float(fpr95),
            "Threshold": float(thr),
        }

        if method in bootstrap_methods:
            ci = bootstrap_ci(id_s, ood_s, seed=seed)
            row.update({
                "AUROC_CI_low": ci["AUROC_CI_low"],
                "AUROC_CI_high": ci["AUROC_CI_high"],
                "FPR95_CI_low": ci["FPR95_CI_low"],
                "FPR95_CI_high": ci["FPR95_CI_high"],
            })
        else:
            row.update({
                "AUROC_CI_low": np.nan,
                "AUROC_CI_high": np.nan,
                "FPR95_CI_low": np.nan,
                "FPR95_CI_high": np.nan,
            })
        rows.append(row)
    return pd.DataFrame(rows)
