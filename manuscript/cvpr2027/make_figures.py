#!/usr/bin/env python3
"""CVPR figures from frozen score/rank CSVs. Does not recompute Kendall W."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "outputs" / "reports"
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

BB_ORDER = [
    "resnet18",
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "effb3",
    "efficientnet_v2_s",
]
BB_LABEL = {
    "resnet18": "R18",
    "resnet50": "R50",
    "densenet121": "Dense",
    "convnext_tiny": "ConvNeXt",
    "mobilenet_v3_large": "Mobile",
    "regnet_y_3_2gf": "RegNet",
    "effb3": "EffB3",
    "efficientnet_v2_s": "EffV2",
    "vit_b_16": "ViT-B/16",
}
METHODS = ["MSP", "Energy", "ELogitNorm", "ViM", "ReAct", "Mahalanobis", "kNN"]
BLUE = "#2c7bb6"
GREY = "#8a8a8a"
ORANGE = "#d95f02"
RED = "#c0392b"
GREEN = "#1a7f4b"

# Official seed-42 MSP (3 d.p. as in the draft) plus ViT held-out.
MSP_CAM = {
    "resnet18": 0.574,
    "resnet50": 0.515,
    "densenet121": 0.883,
    "convnext_tiny": 0.846,
    "mobilenet_v3_large": 0.771,
    "regnet_y_3_2gf": 0.784,
    "effb3": 0.803,
    "efficientnet_v2_s": 0.778,
    "vit_b_16": 0.884,
}
RESNET_MEAN = 0.544654
JUMP_THR = 0.6947

COV_CAM = {  # locked n=4 OOD-triage MSP
    "resnet18": (0.574, 0.145),
    "resnet50": (0.515, 0.149),
    "effb3": (0.803, 0.107),
    "effb3_224": (0.787, 0.013),
}
COV_MID = {
    "resnet18": (0.438, 0.442),
    "resnet50": (0.512, 0.166),
    "effb3": (0.486, 0.712),
}

SEED_MSP = {
    "resnet18": [0.574, 0.712, 0.601, 0.524, 0.581],
    "resnet50": [0.515, 0.621, 0.619, 0.515, 0.661],
    "densenet121": [0.883, 0.865, 0.868, 0.845, 0.826],
    "convnext_tiny": [0.846, 0.799, 0.803, 0.764, 0.785],
    "mobilenet_v3_large": [0.771, 0.728, 0.815, 0.780, 0.745],
    "regnet_y_3_2gf": [0.784, 0.778, 0.836, 0.830, 0.724],
    "effb3": [0.803, 0.731, 0.643, 0.818, 0.678],
    "efficientnet_v2_s": [0.778, 0.744, 0.808, 0.846, 0.835],
}


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1_hook() -> None:
    """Slopegraph R50→DenseNet + coverage cost of copying the AUROC ranking."""
    ranks = pd.read_csv(REP / "architecture_invariance_ranks.csv")
    cam = ranks[ranks.domain == "camelyon17"]
    wide = cam.pivot(index="method", columns="backbone", values="auroc")
    method_style = {
        "Mahalanobis": (GREEN, 2.0, "o"),
        "kNN": ("#2ca25f", 2.0, "s"),
        "MSP": (BLUE, 1.8, "D"),
        "Energy": ("#74add1", 1.2, "^"),
        "ELogitNorm": ("#abd9e9", 1.2, "v"),
        "ReAct": (ORANGE, 1.6, "P"),
        "ViM": (GREY, 1.2, "X"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.55), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    x0, x1 = 0.0, 1.0
    for m, (c, lw, mk) in method_style.items():
        y0, y1 = float(wide.loc[m, "resnet50"]), float(wide.loc[m, "densenet121"])
        ax.plot([x0, x1], [y0, y1], color=c, lw=lw, marker=mk, ms=5, label=m)
        if m in ("Mahalanobis", "kNN", "MSP", "ReAct"):
            ax.text(-0.04, y0, f"{y0:.3f}", ha="right", va="center", fontsize=6, color=c)
            dy = 0.018 if m == "Mahalanobis" else (-0.018 if m == "kNN" else 0.0)
            ax.text(1.04, y1 + dy, f"{y1:.3f}", ha="left", va="center", fontsize=6, color=c)
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(0.48, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ResNet-50", "DenseNet-121"])
    ax.set_ylabel("Camelyon hospital-2 AUROC")
    ax.set_title("(a) Same split, same scores")
    ax.legend(frameon=False, loc="lower right", fontsize=6, ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cov = pd.read_csv(REP / "pathology_risk_coverage/msp_ood_triage_camelyon17_n8.csv")
    order = ["densenet121", "mobilenet_v3_large"]
    labels = ["Select by\nAUROC\n(DenseNet)", "Select by\ncoverage@risk\n(MobileNet)"]
    vals = [float(cov.loc[cov.model == k, "coverage_at_risk10"].iloc[0]) for k in order]
    aurocs = [float(cov.loc[cov.model == k, "ood_auroc"].iloc[0]) for k in order]
    ax = axes[1]
    bars = ax.bar([0, 1], vals, color=[RED, GREEN], edgecolor="black", lw=0.4, width=0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("OOD coverage @ 10% risk")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Cost of copying AUROC rank")
    for i, (v, a) in enumerate(zip(vals, aurocs)):
        ax.text(i, v + 0.03, f"{v:.3f}\nAUROC {a:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig1_hook")


def _rank_matrix(df: pd.DataFrame, domain: str) -> np.ndarray:
    sub = df[df.domain == domain]
    mat = np.zeros((len(METHODS), len(BB_ORDER)))
    for i, m in enumerate(METHODS):
        for j, b in enumerate(BB_ORDER):
            hit = sub[(sub.method == m) & (sub.backbone == b)]
            mat[i, j] = float(hit.iloc[0]["rank"])
    return mat


def fig_rank_heatmaps() -> None:
    df = pd.read_csv(REP / "architecture_invariance_ranks.csv")
    cam = _rank_matrix(df, "camelyon17")
    skin = _rank_matrix(df, "skin_isic_pad")
    cmap = LinearSegmentedColormap.from_list(
        "rank", ["#14532d", "#4ade80", "#fef08a", "#fdba74", "#ef4444"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55), sharey=True)
    for ax, mat, title in (
        (axes[0], cam, "Camelyon17  ($W=0.694$)"),
        (axes[1], skin, "Skin ISIC$\\rightarrow$PAD  ($W=0.791$)"),
    ):
        im = ax.imshow(mat, cmap=cmap, vmin=1, vmax=7, aspect="auto")
        ax.set_xticks(np.arange(len(BB_ORDER)))
        ax.set_xticklabels([BB_LABEL[b] for b in BB_ORDER], rotation=40, ha="right")
        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_yticklabels(METHODS)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = int(mat[i, j])
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="white" if v <= 2 or v >= 6 else "black")
        for i, m in enumerate(METHODS):
            if m in ("Mahalanobis", "kNN"):
                ax.add_patch(Rectangle((-0.5, i - 0.5), len(BB_ORDER), 1,
                                       fill=False, edgecolor="black", lw=0.8))
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=range(1, 8), label="rank (1=best)")
    _save(fig, "fig2_rank_heatmaps")


def fig_coverage() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35))
    # Camelyon n=4
    cam_keys = ["resnet18", "resnet50", "effb3", "effb3_224"]
    cam_lab = ["R18", "R50", "B3@300", "B3@224"]
    x = np.arange(len(cam_keys))
    w = 0.36
    auroc = [COV_CAM[k][0] for k in cam_keys]
    cov = [COV_CAM[k][1] for k in cam_keys]
    axes[0].bar(x - w / 2, auroc, w, color=BLUE, edgecolor="black", lw=0.4, label="MSP AUROC")
    axes[0].bar(x + w / 2, cov, w, color=ORANGE, edgecolor="black", lw=0.4, label="coverage@risk 10%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cam_lab)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Camelyon hospital-2")
    axes[0].set_ylabel("value")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    # MIDOG n=3
    mid_keys = ["resnet18", "resnet50", "effb3"]
    mid_lab = ["R18", "R50", "EffB3"]
    x = np.arange(len(mid_keys))
    auroc = [COV_MID[k][0] for k in mid_keys]
    cov = [COV_MID[k][1] for k in mid_keys]
    axes[1].bar(x - w / 2, auroc, w, color=BLUE, edgecolor="black", lw=0.4, label="MSP AUROC")
    axes[1].bar(x + w / 2, cov, w, color=ORANGE, edgecolor="black", lw=0.4, label="coverage@risk 10%")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mid_lab)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("MIDOG")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    _save(fig, "fig3_coverage_signflip")


def fig_seed_jumps() -> None:
    seeds = np.array([42, 43, 44, 45, 46])
    rmean = np.array([0.545, 0.667, 0.610, 0.520, 0.621])
    jumpers = [
        "densenet121",
        "convnext_tiny",
        "efficientnet_v2_s",
        "mobilenet_v3_large",
        "regnet_y_3_2gf",
        "effb3",
    ]
    fig, ax = plt.subplots(figsize=(6.75, 2.4))
    markers = ["o", "s", "D", "^", "v", "P"]
    colors = ["#14532d", BLUE, "#5e4fa2", "#66c2a5", "#fdae61", GREY]
    for bb, mk, c in zip(jumpers, markers, colors):
        jmp = np.array(SEED_MSP[bb]) - rmean
        ax.plot(seeds, jmp, marker=mk, color=c, lw=1.1, ms=5, label=BB_LABEL[bb])
    ax.axhline(0.15, color=RED, ls="--", lw=0.9)
    ax.text(46.05, 0.158, "bar 0.15", color=RED, fontsize=6.5, va="bottom")
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("MSP $-$ that seed's ResNet mean")
    ax.set_xlim(41.6, 46.6)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "figA_seed_jumps")


if __name__ == "__main__":
    fig1_hook()
    fig_rank_heatmaps()
    fig_coverage()
    fig_seed_jumps()
    print("wrote", list(OUT.glob("fig*.pdf")))
