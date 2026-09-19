#!/usr/bin/env python3
"""Draft (i)+(ii) combined thesis from frozen CSVs. CPU only. No GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DISPLAY = {
    "msp": "MSP",
    "energy": "Energy",
    "mahalanobis": "Mahalanobis",
    "knn": "kNN",
}


def _ranks_high_is_1(values: np.ndarray) -> np.ndarray:
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

REPORTS = Path("outputs/reports")
CAM = REPORTS / "camelyon17/frac1/seed42"
MIDOG = REPORTS / "midog/seed42"
RC = REPORTS / "pathology_risk_coverage/rc_by_model.csv"
W_LOCKED = REPORTS / "architecture_invariance_kendall_w.csv"
OUT_MD = REPORTS / "combined_rank_utility_thesis.md"
OUT_CSV = REPORTS / "combined_rank_utility_table.csv"

CAM_MAP = {
    "resnet18": CAM / "resnet18_score_comparison.csv",
    "resnet50": CAM / "resnet50_score_comparison.csv",
    "effb3": CAM / "effb3_score_comparison.csv",
    "effb3_224": CAM / "effb3_224_score_comparison.csv",
}
MIDOG_MAP = {
    "resnet18": MIDOG / "resnet18_score_comparison.csv",
    "resnet50": MIDOG / "resnet50_score_comparison.csv",  # in-house, not OpenMIBOOD
    "effb3": MIDOG / "effb3_score_comparison.csv",
}
FAMILY = {
    "resnet18": "resnet_like",
    "resnet50": "resnet_like",
    "effb3": "efficientnet_like",
    "effb3_224": "efficientnet_like",
}
LABEL = {
    "resnet18": "ResNet-18@224",
    "resnet50": "ResNet-50@224",
    "effb3": "EffB3@300",
    "effb3_224": "EffB3@224",
}
COMMON = ("msp", "energy", "mahalanobis", "knn")


def load_scores(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(r["Method"]): float(r["AUROC"]) for _, r in df.iterrows()}


def w_from_maps(maps: dict[str, dict[str, float]], methods: tuple[str, ...]) -> float:
    bbs = list(maps)
    mat = np.array([[maps[b][m] for m in methods] for b in bbs], dtype=np.float64)
    ranks = np.vstack([_ranks_high_is_1(row) for row in mat])
    return kendall_w(ranks)


def msp_rank(scores: dict[str, float], methods: tuple[str, ...]) -> int:
    vals = np.array([scores[m] for m in methods], dtype=np.float64)
    ranks = _ranks_high_is_1(vals)
    return int(ranks[methods.index("msp")])


def top1(scores: dict[str, float], methods: tuple[str, ...]) -> str:
    return max(methods, key=lambda m: scores[m])


def main() -> None:
    cam = {b: load_scores(p) for b, p in CAM_MAP.items()}
    mid = {b: load_scores(p) for b, p in MIDOG_MAP.items()}
    w_cam4 = w_from_maps(cam, COMMON)
    w_mid3 = w_from_maps(mid, COMMON)
    w_locked = pd.read_csv(W_LOCKED).set_index("domain")["kendall_w"].to_dict()

    rc = pd.read_csv(RC)
    rc_msp = rc[(rc["protocol"] == "ood_triage") & (rc["method"] == "msp")]
    rc_msp = rc_msp.set_index(["domain", "model"])

    def rc_row(domain, b, scores):
        key = (domain, b)
        return {
            "domain": domain,
            "backbone": b,
            "label": LABEL[b],
            "family": FAMILY[b],
            "msp_auroc": scores["msp"],
            "energy_auroc": scores["energy"],
            "mahalanobis_auroc": scores["mahalanobis"],
            "knn_auroc": scores["knn"],
            "msp_rank_among_msp_energy_maha_knn": msp_rank(scores, COMMON),
            "top1_among_msp_energy_maha_knn": DISPLAY[top1(scores, COMMON)],
            "ood_triage_msp_coverage_at_risk10": float(rc_msp.loc[key, "coverage_at_risk10"]),
            "ood_triage_msp_aurc": float(rc_msp.loc[key, "aurc"]),
            "ood_triage_msp_full_risk": float(rc_msp.loc[key, "full_risk"]),
        }

    rows = [rc_row("camelyon17", b, s) for b, s in cam.items()]
    rows += [rc_row("midog", b, s) for b, s in mid.items()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    cam_df = df[df["domain"] == "camelyon17"]
    mid_df = df[df["domain"] == "midog"]

    cam_by_auroc = " > ".join(cam_df.sort_values("msp_auroc", ascending=False)["label"])
    cam_by_cov = " > ".join(
        cam_df.sort_values("ood_triage_msp_coverage_at_risk10", ascending=False)["label"]
    )
    mid_by_auroc = " > ".join(mid_df.sort_values("msp_auroc", ascending=False)["label"])
    mid_by_cov = " > ".join(
        mid_df.sort_values("ood_triage_msp_coverage_at_risk10", ascending=False)["label"]
    )
    winner_auroc = cam_df.loc[cam_df["msp_auroc"].idxmax()]
    winner_cov = cam_df.loc[cam_df["ood_triage_msp_coverage_at_risk10"].idxmax()]
    loser_cov = cam_df.loc[cam_df["ood_triage_msp_coverage_at_risk10"].idxmin()]
    mid_auroc_w = mid_df.loc[mid_df["msp_auroc"].idxmax()]
    mid_cov_w = mid_df.loc[mid_df["ood_triage_msp_coverage_at_risk10"].idxmax()]
    mid_cov_l = mid_df.loc[mid_df["ood_triage_msp_coverage_at_risk10"].idxmin()]

    md = f"""# Combined thesis: rank instability (i) + utility mismatch (ii)

Locked 2026-09-17. CPU-only draft from existing CSVs. No new GPU.
Does not wait for zoo `59493` or remaining FM cells. Kendall's W is **per domain**, never pooled.

Phao A is **not** in this draft as a headline. Scope: FM amplifies a Maha–MSP gap that already exists on CNN; upgrade only if GigaPath-Camelyon or Phikon-Camelyon invert (MSP AUROC < 0.5, sign-check `AUROC + AUROC_flipped = 1`). Until then: MIDOG-specific footnote (`mismatch=severe`).

## Core claim

OOD evaluation is wrong in two independent ways at once:

1. **Rank instability (cửa 2).** Which method looks best depends on the backbone. Founding Camelyon cell: MSP AUROC 0.51–0.57 on ResNet-18/50 vs **0.803** on EffB3@300, and **0.787** on EffB3@224 (resolution control — not a 300px artifact).
2. **Utility mismatch is domain-dependent unpredictability, not a universal reversal.** AUROC rank and OOD-triage coverage@risk10% rank **disagree on both domains**, but **the direction of the disagreement flips**. Do not claim “AUROC-winner = utility-loser” as a law — that pairing is Camelyon-specific (n=1 domain) and is exactly the single-domain overclaim that desk-rejected the old CMPB triage paradox.

Actionable warning: do not pick backbone/method from an AUROC leaderboard for triage; check coverage@risk on the target domain. The sign of the AUROC↔coverage mismatch is **not predictable** from Camelyon alone.

## Trục (i) — Kendall's W (not pooled)

Locked n=3, 7 methods (MSP, Energy, ELogitNorm, ViM, ReAct, Maha, kNN), from `architecture_invariance_kendall_w.csv`:

| domain | n_backbones | Kendall's W |
|---|---|---|
| Camelyon17 | 3 (R18, R50, EffB3@300) | {w_locked['camelyon17']:.3f} |
| MIDOG | 3 (R18, R50 in-house, EffB3@300) | {w_locked['midog']:.3f} |
| Skin ISIC→PAD | 3 | {w_locked['skin_isic_pad']:.3f} |

W=1 identical method ranking across backbones; W=0 none. Camelyon W={w_locked['camelyon17']:.3f} is the cửa-2 number. Do not average with MIDOG/skin.

Same four common methods (MSP, Energy, Maha, kNN) after adding EffB3@224:

| domain | backbones | Kendall's W |
|---|---|---|
| Camelyon17 | R18, R50, EffB3@300, EffB3@224 | {w_cam4:.3f} |
| MIDOG | R18, R50 in-house, EffB3@300 | {w_mid3:.3f} |

On Camelyon the top-1 method among those four is **Mahalanobis** on both ResNets and EffB3@224, but **kNN** on EffB3@300. That is the rank flip. MSP AUROC still jumps on both EffB3 resolutions.

MIDOG R50 uses **in-house** `midog/seed42/resnet50_score_comparison.csv` (MSP 0.512), not OpenMIBOOD 0.59.

## Trục (ii) — OOD-triage coverage@risk 10% (MSP)

Protocol: rank OOD samples by MSP (higher = more ID-like), measure classifier error on the accepted prefix, take the largest coverage whose risk ≤ 10%. Same script for both domains (`pathology_risk_coverage.py`). ID-selective Camelyon is **vacuous** (ID-val acc ≈ 99.5%, coverage@risk10% = 1.0 for every backbone). Do not cite it.

### Camelyon (hospital-2)

| backbone | family | MSP AUROC | MSP rank among {{MSP,Energy,Maha,kNN}} (1=best) | top-1 | coverage@risk10% | AURC (lower better) |
|---|---|---|---|---|---|---|
"""
    for _, r in cam_df.iterrows():
        md += (
            f"| {r['label']} | {r['family']} | {r['msp_auroc']:.3f} | "
            f"{int(r['msp_rank_among_msp_energy_maha_knn'])} | "
            f"{r['top1_among_msp_energy_maha_knn']} | "
            f"{r['ood_triage_msp_coverage_at_risk10']:.3f} | "
            f"{r['ood_triage_msp_aurc']:.3f} |\n"
        )

    md += f"""
AUROC order: **{cam_by_auroc}**
coverage@risk10% order: **{cam_by_cov}**

On Camelyon, AUROC winner **{winner_auroc['label']}** (MSP {winner_auroc['msp_auroc']:.3f}) is not the coverage@risk10% winner (**{winner_cov['label']}**, {winner_cov['ood_triage_msp_coverage_at_risk10']:.3f}). Coverage loser: **{loser_cov['label']}** ({loser_cov['ood_triage_msp_coverage_at_risk10']:.3f}). This is an existence proof on one domain, not a law.

### MIDOG (in-house R18 / R50 / EffB3@300) — direction does not copy Camelyon

Protocol identical. n_OOD=5902. ID n_val=251 is small; MSP AUROC spread is tiny (0.438–0.512). Do not over-read “winner” labels here. Full table, not a single EffB3 cell:

| backbone | family | MSP AUROC | coverage@risk10% |
|---|---|---|---|
"""
    for _, r in mid_df.iterrows():
        md += (
            f"| {r['label']} | {r['family']} | {r['msp_auroc']:.3f} | "
            f"{r['ood_triage_msp_coverage_at_risk10']:.3f} |\n"
        )

    md += f"""
AUROC order: **{mid_by_auroc}**
coverage@risk10% order: **{mid_by_cov}**

AUROC-highest: **{mid_auroc_w['label']}** ({mid_auroc_w['msp_auroc']:.3f}).
coverage@risk10% highest: **{mid_cov_w['label']}** ({mid_cov_w['ood_triage_msp_coverage_at_risk10']:.3f}).
coverage@risk10% lowest: **{mid_cov_l['label']}** ({mid_cov_l['ood_triage_msp_coverage_at_risk10']:.3f}).

EffB3 does **not** win both on MIDOG. It wins coverage (0.712) but not AUROC (0.486 vs R50 0.512). R50 is the AUROC-highest backbone and the coverage-lowest (0.166). That is still not a license to write “AUROC-winner = utility-loser” as a universal rule: MIDOG AUROC gaps are small, and the **permutation of ranks is not the Camelyon permutation**.

**Claim that survives both tables:** the AUROC↔coverage mapping is **unreliable**, and **the direction of unreliability is domain-dependent**. Camelyon: EffNet tops AUROC and sits at the bottom of coverage. MIDOG: the ranking permutes the other way (R50 tops AUROC / bottoms coverage; EffB3 bottoms-to-mid AUROC / tops coverage). This is instability stacked on cửa 2, not a second copy of the Camelyon reversal. Reviewer-safe sentence: *do not choose a backbone for triage from an AUROC leaderboard; measure coverage@risk on the deployment domain, because the sign of the mismatch cannot be predicted from Camelyon.*

Do not pool Kendall's W with these coverage ranks.

## What this is not

- Not “EffB3 always loses triage.” False on MIDOG.
- Not a frozen-FM headline (Phao A). UNI-Camelyon MSP is 0.54–0.60, not invert. CNN R18-MIDOG MSP is already 0.438.
- Not zoo A/B/C. n=4 Camelyon spans 2 families; A/B/C still waits for n=8 (`59493`). After zoo, test whether **unpredictability** (rank disagreement) spans families — not whether EffB3-as-utility-loser does.
- Not ID-val risk-coverage. That protocol saturates on Camelyon and on skin actually favored EffB3. The mismatch is **OOD-triage**.

## Next (no GPU now)

2. Zoo `59493` done → rerun OOD-triage coverage@risk10% on all 8 Camelyon backbones (add DenseNet121, ConvNeXt-Tiny, MobileNetV3-L, RegNetY-3.2GF, EffV2-S@384). Question: does AUROC↔coverage **disagreement** span families, not “does EffB3 always lose.”
3. GigaPath-Camelyon + Phikon done → rerun `scripts/audit_fm_msp_sign.py`. Invert on Camelyon → upgrade Phao A. Else keep MIDOG-only footnote.
"""

    OUT_MD.write_text(md)
    print(f"Wrote {OUT_CSV}")
    print(df.to_string(index=False))
    print(f"W camelyon n=4 common4={w_cam4:.3f}  midog n=3 common4={w_mid3:.3f}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
