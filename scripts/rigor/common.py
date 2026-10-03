#!/usr/bin/env python3
"""Shared helpers for the rigor pack.

Python 3.8 compatible. Does NOT import torch at module level, so the CSV-only
stages run on a plain numpy/pandas(/scipy) Python.

Kendall's W and the rank convention are imported from
scripts/rebuild_architecture_invariance.py when that import works (it pulls
in torchvision through src.models.cnn_family). If it does not, a verbatim
copy of the same two functions is used; smoke_test.py asserts both agree.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ----------------------------------------------------------------------------
# Constants (same names/order as rebuild_architecture_invariance.py)
# ----------------------------------------------------------------------------
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
METHOD_KEYS = list(DISPLAY.keys())
LOGIT_METHODS = ("MSP", "Energy", "ELogitNorm")
FEATURE_METHODS = ("Mahalanobis", "kNN")

# Official 8-CNN zoo. Names as in architecture_invariance_ranks.csv.
ARCHS = (
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "effb3",
    "efficientnet_v2_s",
)
ANCHORS = ("resnet18", "resnet50", "densenet121", "effb3")
RESNETS = ("resnet18", "resnet50")
SEEDS = (42, 43, 44, 45, 46)
EXTRA_SEEDS = (43, 44, 45, 46)
SHORT = {
    "resnet18": "R18",
    "resnet50": "R50",
    "densenet121": "DenseNet",
    "convnext_tiny": "ConvNeXt",
    "mobilenet_v3_large": "MobileNet",
    "regnet_y_3_2gf": "RegNet",
    "effb3": "EffB3",
    "efficientnet_v2_s": "EffV2-S",
    "vit_b_16": "ViT-B/16",
}
DOMAINS = ("camelyon17", "skin_isic_pad", "midog")
JUMP_DELTA = 0.15
RISKS = (0.05, 0.10, 0.20)
TARGET = 0.10

# Published numbers used ONLY as reproduction checks (never as inputs).
PUBLISHED = {
    "official_w_camelyon": 0.694,
    "official_w_skin": 0.791,
    "cross_seed_w": {"resnet50": 0.931, "resnet18": 0.800, "densenet121": 0.789, "effb3": 0.600},
    "anchor4_cross_arch_w": {42: 0.750, 43: 0.786, 44: 0.772, 45: 0.781, 46: 0.915},
    "frozen_split": {"auroc_pick": ("convnext_tiny", 0.539), "cov_pick": ("efficientnet_v2_s", 0.754),
                     "best": ("mobilenet_v3_large", 1.000)},
    "oracle_full": {"densenet121": 0.078, "mobilenet_v3_large": 0.904},
}


def file_stem(arch: str, domain: str = "camelyon17") -> str:
    """Artifact stem used on disk. Official EffV2-S cell is the @224 run."""
    if arch == "efficientnet_v2_s":
        return "efficientnet_v2_s_224"
    return arch


# ----------------------------------------------------------------------------
# Kendall W (reuse the repo's implementation)
# ----------------------------------------------------------------------------
def _ranks_high_is_1_copy(values: np.ndarray) -> np.ndarray:
    """Verbatim copy of rebuild_architecture_invariance._ranks_high_is_1."""
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


def _kendall_w_copy(rank_matrix: np.ndarray) -> float:
    """Verbatim copy of rebuild_architecture_invariance.kendall_w."""
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


def _load_rebuild():
    try:
        spec = importlib.util.spec_from_file_location(
            "rebuild_architecture_invariance", REPO / "scripts" / "rebuild_architecture_invariance.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # torch/torchvision missing on system python
        return None


_REBUILD = _load_rebuild()
if _REBUILD is not None:
    kendall_w = _REBUILD.kendall_w
    ranks_high_is_1 = _REBUILD._ranks_high_is_1
    W_SOURCE = "scripts/rebuild_architecture_invariance.py"
else:
    kendall_w = _kendall_w_copy
    ranks_high_is_1 = _ranks_high_is_1_copy
    W_SOURCE = "verbatim copy in scripts/rigor/common.py (rebuild import failed)"


def ranks_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise ranks, 1 = highest value (same as rebuild._ranks_high_is_1_rows)."""
    mat = np.asarray(mat, dtype=np.float64)
    out = np.empty_like(mat)
    for i in range(len(mat)):
        out[i] = ranks_high_is_1(mat[i])
    return out


def tie_terms(ranks: np.ndarray) -> np.ndarray:
    """Per-row sum(t^3 - t) over tie groups. ranks: (..., n)."""
    flat = ranks.reshape(-1, ranks.shape[-1])
    out = np.empty(len(flat))
    for i, row in enumerate(flat):
        _, c = np.unique(row, return_counts=True)
        out[i] = float(np.sum(c**3 - c))
    return out.reshape(ranks.shape[:-1])


def kendall_w_batch(ranks: np.ndarray, ties: Optional[np.ndarray] = None) -> np.ndarray:
    """Vectorised tie-corrected W. ranks: (..., k, n). ties: (..., k) per-row tie terms."""
    r = np.asarray(ranks, dtype=np.float64)
    k, n = r.shape[-2], r.shape[-1]
    col = r.sum(axis=-2)
    s = np.sum((col - col.sum(axis=-1, keepdims=True) / n) ** 2, axis=-1)
    if ties is None:
        tsum = 0.0
    else:
        tsum = np.asarray(ties, dtype=np.float64).sum(axis=-1)
    denom = k**2 * (n**3 - n) - k * tsum
    return 12.0 * s / denom


def friedman_from_w(w: float, k: int, n: int) -> Tuple[float, float]:
    """Friedman chi2 = k(n-1)W, df = n-1 (tie-corrected because W is)."""
    chi2 = float(k * (n - 1) * w)
    try:
        from scipy.stats import chi2 as _chi2

        p = float(_chi2.sf(chi2, n - 1))
    except Exception:
        p = float("nan")
    return chi2, p


def perm_p(observed: float, null: np.ndarray, alternative: str = "greater") -> float:
    """Same convention as scripts/kendall_w_perm.py: (hits + 1) / (N + 1)."""
    null = np.asarray(null)
    if alternative == "greater":
        hits = int(np.sum(null >= observed - 1e-12))
    else:
        hits = int(np.sum(null <= observed + 1e-12))
    return float((hits + 1) / (len(null) + 1))


def w_perm_null(ranks: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Null of W: independently permute each rater's (row's) ranks. Keeps tie structure."""
    k, n = ranks.shape
    ties = tie_terms(ranks)
    idx = np.argsort(rng.random((n_perm, k, n)), axis=-1)
    shuf = np.take_along_axis(np.broadcast_to(ranks, (n_perm, k, n)), idx, axis=-1)
    return kendall_w_batch(shuf, np.broadcast_to(ties, (n_perm, k)))


# ----------------------------------------------------------------------------
# Multiple testing
# ----------------------------------------------------------------------------
def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def bh(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)[::-1]
    adj = np.empty(m)
    run = 1.0
    for pos, i in enumerate(order):
        rank = m - pos
        run = min(run, pv[i] * m / rank)
        adj[i] = run
    out[ok] = adj
    return out


# ----------------------------------------------------------------------------
# Score-CSV access (tracked in git): AUROC / FPR95 matrices
# ----------------------------------------------------------------------------
def ranks_file(root: Path) -> Path:
    return root / "outputs/reports/architecture_invariance_ranks.csv"


def score_csv_path(root: Path, domain: str, arch: str, seed: int) -> Path:
    stem = file_stem(arch, domain)
    rep = root / "outputs/reports"
    if domain == "camelyon17":
        return rep / "camelyon17/frac1" / ("seed%d" % seed) / ("%s_score_comparison.csv" % stem)
    if domain == "skin_isic_pad":
        if seed == 42:
            if arch in ("resnet18", "resnet50", "effb3"):
                return rep / ("%s_score_comparison.csv" % arch)
            return rep / "skin" / ("%s_score_comparison.csv" % stem)
        return rep / "skin" / ("seed%d" % seed) / ("%s_score_comparison.csv" % stem)
    if domain == "midog":
        return rep / "midog" / ("seed%d" % seed) / ("%s_score_comparison.csv" % stem)
    raise ValueError(domain)


def metric_vector(root: Path, domain: str, arch: str, seed: int, metric: str = "AUROC") -> np.ndarray:
    """7-score vector in METHODS_ORDER. Seed-42 AUROC comes from the frozen ranks file
    (same convention as scripts/camelyon_cross_seed_w.py); other cells from score CSVs.
    Missing methods -> NaN."""
    if metric == "AUROC" and seed == 42 and ranks_file(root).exists():
        df = pd.read_csv(ranks_file(root))
        sub = df[(df.domain == domain) & (df.backbone == arch)].set_index("method")["auroc"]
        if len(sub):
            return np.array([float(sub.get(m, np.nan)) for m in METHODS_ORDER])
    path = score_csv_path(root, domain, arch, seed)
    if not path.exists():
        return np.full(len(METHODS_ORDER), np.nan)
    df = pd.read_csv(path)
    if metric not in df.columns:
        return np.full(len(METHODS_ORDER), np.nan)
    by = {DISPLAY[m]: float(a) for m, a in zip(df["Method"], df[metric]) if m in DISPLAY}
    return np.array([by.get(m, np.nan) for m in METHODS_ORDER])


def load_cube(root: Path, domain: str, archs: Sequence[str], seeds: Sequence[int],
              metric: str = "AUROC") -> np.ndarray:
    """(n_arch, n_seed, 7). NaN where a cell/method is missing."""
    return np.stack([np.stack([metric_vector(root, domain, a, s, metric) for s in seeds]) for a in archs])


# ----------------------------------------------------------------------------
# Coverage@risk (reuse) + weighted versions for bootstraps / subsets
# ----------------------------------------------------------------------------
def coverage_at_risk(scores: np.ndarray, correct: np.ndarray, target: float = TARGET) -> float:
    """Same as scripts/camelyon_coverage_val_split.coverage_at_risk (copied there and in
    bootstrap_coverage_ci.py). Kept here so CPU stages do not import torch."""
    order = np.argsort(-scores, kind="mergesort")
    csort = correct[order]
    k = np.arange(1, csort.size + 1, dtype=np.float64)
    ok = (1.0 - np.cumsum(csort) / k) <= target + 1e-12
    if not np.any(ok):
        return 0.0
    return float(np.nonzero(ok)[0][-1] + 1) / csort.size


class CoverageSorter(object):
    """Pre-sorts one OOD score vector so weighted coverage@risk is O(n) per call.
    With unit weights equals coverage_at_risk; with 0/1 weights equals coverage on
    the subset (mergesort is stable, so ties keep index order)."""

    def __init__(self, scores: np.ndarray, correct: np.ndarray):
        self.order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
        self.c = np.asarray(correct, dtype=np.float64)[self.order]

    def coverage(self, weights: Optional[np.ndarray] = None, target: float = TARGET) -> float:
        if weights is None:
            w = np.ones_like(self.c)
        else:
            w = np.asarray(weights, dtype=np.float64)[self.order]
        cw = np.cumsum(w)
        tot = cw[-1]
        if tot <= 0:
            return float("nan")
        cc = np.cumsum(w * self.c)
        with np.errstate(divide="ignore", invalid="ignore"):
            risk = 1.0 - cc / cw
        ok = (cw > 0) & (risk <= target + 1e-12) & (w > 0)
        if not np.any(ok):
            return 0.0
        return float(cw[np.nonzero(ok)[0][-1]] / tot)


class AurocSorter(object):
    """Pre-groups ID+OOD scores so weighted AUROC is O(n) per call.
    Scores are cast to float32 first, exactly like
    src.utils.benchmark_metrics.calc_auroc (sklearn roc_auc_score on float32)."""

    def __init__(self, id_scores: np.ndarray, ood_scores: np.ndarray):
        s = np.concatenate([np.asarray(id_scores, dtype=np.float32), np.asarray(ood_scores, dtype=np.float32)])
        self.n_id = len(id_scores)
        self.is_id = np.zeros(len(s), dtype=bool)
        self.is_id[: self.n_id] = True
        _, self.g = np.unique(s, return_inverse=True)
        self.ng = int(self.g.max()) + 1

    def auroc(self, w_id: Optional[np.ndarray] = None, w_ood: Optional[np.ndarray] = None) -> float:
        n_ood = len(self.is_id) - self.n_id
        wi = np.ones(self.n_id) if w_id is None else np.asarray(w_id, dtype=np.float64)
        wo = np.ones(n_ood) if w_ood is None else np.asarray(w_ood, dtype=np.float64)
        gid = np.bincount(self.g[: self.n_id], weights=wi, minlength=self.ng)
        good = np.bincount(self.g[self.n_id :], weights=wo, minlength=self.ng)
        below = np.cumsum(good) - good
        den = gid.sum() * good.sum()
        if den <= 0:
            return float("nan")
        return float(np.sum(gid * (below + 0.5 * good)) / den)


# ----------------------------------------------------------------------------
# DeLong (fast, Sun & Xu 2014)
# ----------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x, kind="mergesort")
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n, dtype=np.float64)
    out[j] = t
    return out


def delong(id_scores_list: Sequence[np.ndarray], ood_scores_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """AUROCs (ID = positive class, higher score = more ID) and their covariance,
    for several scores computed on the SAME samples (paired)."""
    pos = np.stack([np.asarray(s, dtype=np.float32).astype(np.float64) for s in id_scores_list])
    neg = np.stack([np.asarray(s, dtype=np.float32).astype(np.float64) for s in ood_scores_list])
    m, n = pos.shape[1], neg.shape[1]
    k = pos.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def norm_sf(z: float) -> float:
    from math import erfc, sqrt

    return 0.5 * erfc(z / sqrt(2.0))


# ----------------------------------------------------------------------------
# Feature caches / metadata (HPC paths, never in git)
# ----------------------------------------------------------------------------
def feature_path(feat_root: Path, domain: str, arch: str, seed: int, indexed: bool = False) -> Path:
    stem = file_stem(arch, domain)
    if domain == "camelyon17":
        d = feat_root / "outputs/features/camelyon17/frac1" / ("seed%d" % seed)
        if indexed:
            d = d / "rigor_indexed"
        return d / ("%s_features.pt" % stem)
    if domain == "skin_isic_pad":
        if seed == 42:
            if arch in ("resnet18", "resnet50", "effb3"):
                return feat_root / "outputs/features" / ("%s_isic_pad_features.pt" % arch)
            return feat_root / "outputs/features/skin" / ("%s_isic_pad_features.pt" % stem)
        return feat_root / "outputs/features/skin" / ("seed%d" % seed) / ("%s_isic_pad_features.pt" % stem)
    if domain == "midog":
        return feat_root / "outputs/features/midog" / ("seed%d" % seed) / ("%s_features.pt" % stem)
    raise ValueError(domain)


def score_cache_path(out_root: Path, domain: str, arch: str, seed: int) -> Path:
    return out_root / "scores" / domain / ("seed%d" % seed) / ("%s.npz" % file_stem(arch, domain))


def default_out(root: Path) -> Path:
    """Large intermediate files (score caches): outputs/rigor_pack (gitignored)."""
    return root / "outputs/rigor_pack"


def default_reports(root: Path) -> Path:
    """Small result tables (tracked like every other report): outputs/reports/rigor_pack."""
    return root / "outputs/reports/rigor_pack"


def metadata_path(feat_root: Path) -> Path:
    return feat_root / "data/raw/wilds/camelyon17_v1.0/metadata.csv"


def load_camelyon_metadata(feat_root: Path) -> Optional[pd.DataFrame]:
    """WILDS metadata with the official split applied (center 1 -> val=3, center 2 -> test=2),
    in WILDS row order (= dataset index). Returns None if not available."""
    p = metadata_path(feat_root)
    if not p.exists():
        return None
    meta = pd.read_csv(p, index_col=0, dtype={"patient": str})
    meta = meta.reset_index(drop=True)
    split = meta["split"].to_numpy().copy()
    split[meta["center"].to_numpy() == 1] = 3
    split[meta["center"].to_numpy() == 2] = 2
    meta["wilds_split"] = split  # 0 train, 1 id_val, 2 test (hospital 2), 3 val (hospital 1)
    return meta


def frozen_val_patients(root: Path) -> Optional[List[str]]:
    p = root / "outputs/reports/camelyon_coverage_val_split_patients.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={"patient": str})
    return sorted(df.loc[df.half == "val", "patient"].tolist())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_text(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    text = "\n".join(lines) + "\n"
    path.write_text(text)


def env_info() -> Dict[str, str]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "w_source": W_SOURCE}


def percentile_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def all_subsets(items: Sequence[str], k: int) -> List[Tuple[str, ...]]:
    return list(itertools.combinations(items, k))


def dump_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n")
