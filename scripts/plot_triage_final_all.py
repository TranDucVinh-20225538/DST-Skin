# scripts/run_risk_coverage_backbones.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.feature_extractor import extract_features_and_logits
from src.utils.scoring import OODScorer
from src.datasets.isic_dataset import ISICDataset
from src.models.resnet18 import get_resnet18
from src.models.resnet50 import get_resnet50
from src.models.efficientnet_b3 import get_efficientnet_b3


# ====================== CONFIG ======================
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
BATCH_SIZE = 32
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def get_trained_scorer(model, train_loader, device):
    """Fit OODScorer on ID training features for a given backbone."""
    print(f"--- Fitting scorer for {model.__class__.__name__} ---")
    _, train_feats = extract_features_and_logits(model, train_loader, device)
    scorer = OODScorer(k_nearest=50)
    scorer.fit(train_feats)
    return scorer


def calculate_risk_coverage(model, loader, scorer, device):
    """
    Compute risk--coverage on PAD-UFES.
    Samples are ranked by Mahalanobis-based reliability
    (more ID-like / more reliable first).
    Coverage decreases from 100% to 10%.
    """
    model.eval()
    logits, feats = extract_features_and_logits(model, loader, device)

    # NOTE: đảm bảo score_mahalanobis trả về "cao = ID-like hơn".
    # Nếu nó trả về distance (cao = OOD hơn), bạn nên đổi thành:
    # scores = -scorer.score_mahalanobis(scorer.l2_normalize(feats))
    scores = scorer.score_mahalanobis(scorer.l2_normalize(feats))
    preds = np.argmax(logits, axis=1)

    # Lấy labels từ dataset, bỏ qua các field khác (ví dụ metadata)
    all_labels = []
    for batch in loader:
        # Nếu dataset trả về (x, y) => len(batch) == 2
        # Nếu (x, y, meta) => len(batch) == 3
        if len(batch) >= 2:
            y = batch[1]
        else:
            raise ValueError("Unexpected batch format from loader")
        all_labels.append(y.numpy())
    labels = np.concatenate(all_labels)

    # Sắp xếp theo độ tin cậy giảm dần (ID-like cao trước)
    idx = np.argsort(-scores)
    sorted_labels = labels[idx]
    sorted_preds = preds[idx]

    coverages = np.linspace(1.0, 0.1, 100)
    risks = []
    n_total = len(sorted_labels)

    for c in coverages:
        n = max(1, int(c * n_total))
        error_rate = 1.0 - np.mean(sorted_preds[:n] == sorted_labels[:n])
        risks.append(error_rate)

    return coverages * 100, np.array(risks)


def main():
    print(f"Running clinical triage risk--coverage analysis on {DEVICE}...")

    tfm_224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    tfm_300 = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_csv = "data/processed/isic2018_binary/isic_2018_binary_train.csv"
    train_img = "data/raw/isic2018/ISIC2018_Task3_Training_Input"
    ood_csv   = "data/processed/pad_ufes20_binary/pad_ufes_binary.csv"
    ood_img   = "data/raw/pad_ufes20/images"

    # ===== 1. ResNet-18 =====
    print("\n[1/3] Processing ResNet-18...")
    m18 = get_resnet18(num_classes=2).to(DEVICE)
    m18.load_state_dict(torch.load(
        "data/models/resnet18_robust_epoch10.pth",
        map_location=DEVICE
    ))

    train_loader_18 = DataLoader(
        ISICDataset(train_csv, train_img, transform=tfm_224),
        batch_size=BATCH_SIZE, shuffle=False
    )
    s18 = get_trained_scorer(m18, train_loader_18, DEVICE)

    ood_loader_18 = DataLoader(
        ISICDataset(ood_csv, ood_img, transform=tfm_224),
        batch_size=BATCH_SIZE, shuffle=False
    )
    cov18, risk18 = calculate_risk_coverage(m18, ood_loader_18, s18, DEVICE)

    # ===== 2. ResNet-50 =====
    print("\n[2/3] Processing ResNet-50...")
    m50 = get_resnet50(num_classes=2).to(DEVICE)
    m50.load_state_dict(torch.load(
        "data/models/resnet50_robust_best.pth",
        map_location=DEVICE
    ))

    s50 = get_trained_scorer(m50, train_loader_18, DEVICE)
    cov50, risk50 = calculate_risk_coverage(m50, ood_loader_18, s50, DEVICE)

    # ===== 3. EfficientNet-B3 =====
    print("\n[3/3] Processing EfficientNet-B3...")
    mb3 = get_efficientnet_b3(num_classes=2).to(DEVICE)
    mb3.load_state_dict(torch.load(
        "data/models/efficientnet_b3_robust_best.pth",
        map_location=DEVICE
    ))

    train_loader_300 = DataLoader(
        ISICDataset(train_csv, train_img, transform=tfm_300),
        batch_size=BATCH_SIZE, shuffle=False
    )
    sb3 = get_trained_scorer(mb3, train_loader_300, DEVICE)

    ood_loader_300 = DataLoader(
        ISICDataset(ood_csv, ood_img, transform=tfm_300),
        batch_size=BATCH_SIZE, shuffle=False
    )
    covb3, riskb3 = calculate_risk_coverage(mb3, ood_loader_300, sb3, DEVICE)

    # ===== PLOT =====
    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.plot(
        cov18, risk18, label="ResNet-18",
        color="#2E86DE", lw=2.6, marker="o", markersize=2.5, markevery=8
    )
    ax.plot(
        cov50, risk50, label="ResNet-50",
        color="#E67E22", lw=2.6, marker="s", markersize=2.5, markevery=8
    )
    ax.plot(
        covb3, riskb3, label="EfficientNet-B3",
        color="#E74C3C", lw=3.0, marker="^", markersize=2.8, markevery=8
    )

    ax.axhline(
        y=0.15, color="dimgray", linestyle="--", lw=1.8, alpha=0.9,
        label="Clinical risk threshold (15%)"
    )

    ax.set_title("Risk--coverage under ISIC$\\rightarrow$PAD-UFES shift", pad=12, weight="bold")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Risk (empirical error rate)")
    ax.set_xlim(100, 10)
    ax.set_ylim(0.0, 0.6)

    ax.tick_params(axis="both", which="major", length=5, width=1)
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.65)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.8
    )
    legend.get_frame().set_edgecolor("0.8")

    fig.tight_layout()

    png_path = f"{OUTPUT_DIR}/SOTA_TRIAGE_COMPARISON_FINAL.png"
    pdf_path = f"{OUTPUT_DIR}/SOTA_TRIAGE_COMPARISON_FINAL.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to:\n- {png_path}\n- {pdf_path}")


if __name__ == "__main__":
    main()