import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.utils.feature_extractor import extract_features_and_logits
from src.datasets.isic_dataset import ISICDataset
from src.models.efficientnet_b3 import get_efficientnet_b3

# ====================== CONFIG ======================
SEED = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
MODEL_PATH = "data/models/efficientnet_b3_robust_best.pth"
OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset paths
ISIC_CSV = "data/processed/isic2018_binary/isic_2018_binary_val.csv"
ISIC_IMG_DIR = "data/raw/isic2018/ISIC2018_Task3_Validation_Input"

PAD_CSV = "data/processed/pad_ufes20_binary/pad_ufes_binary.csv"
PAD_IMG_DIR = "data/raw/pad_ufes20/images"

# UMAP config
N_NEIGHBORS = 50
MIN_DIST = 0.3
UMAP_METRIC = "cosine"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    print(f"🚀 Starting UMAP visualization on {DEVICE}...")

    # 1. Transform chuẩn cho EfficientNet-B3 (300x300)
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 2. Load model EfficientNet-B3 đã train
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

    print(f"📦 Loading model from {MODEL_PATH}...")
    model = get_efficientnet_b3(num_classes=2).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load dữ liệu
    print("📂 Loading datasets...")

    # In-distribution: ISIC Val (giới hạn 1000 mẫu để UMAP nhanh)
    val_ds = ISICDataset(ISIC_CSV, ISIC_IMG_DIR, transform=transform)
    num_id_samples = min(1000, len(val_ds))
    id_indices = np.random.choice(len(val_ds), num_id_samples, replace=False)
    val_subset = Subset(val_ds, id_indices)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Out-of-distribution: toàn bộ PAD-UFES
    ood_ds = ISICDataset(PAD_CSV, PAD_IMG_DIR, transform=transform)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"   • ISIC (ID) samples used: {num_id_samples}")
    print(f"   • PAD-UFES (OOD) samples: {len(ood_ds)}")

    # 4. Trích xuất đặc trưng (features)
    print("🧠 Extracting features from backbone...")
    # Chỉ lấy features, bỏ logits
    _, val_feats = extract_features_and_logits(model, val_loader, DEVICE)
    _, ood_feats = extract_features_and_logits(model, ood_loader, DEVICE)

    all_feats = np.concatenate([val_feats, ood_feats], axis=0)

    # Nhãn để vẽ (0: ISIC, 1: PAD-UFES)
    labels = np.concatenate([
        np.zeros(len(val_feats), dtype=int),
        np.ones(len(ood_feats), dtype=int),
    ])
    dataset_names = np.where(
        labels == 0,
        "ISIC (In-Distribution)",
        "PAD-UFES (Out-of-Distribution)"
    )

    print(f"📉 Running UMAP on {all_feats.shape[0]} samples "
          f"with {all_feats.shape[1]} dims...")
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
    )
    embedding = reducer.fit_transform(all_feats)
    print("✅ UMAP reduction complete.")

    # 6. Vẽ biểu đồ Seaborn
    print("🎨 Plotting UMAP scatter...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    ax = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=dataset_names,
        palette=["#3498db", "#e74c3c"],  # Xanh dương (ID), Đỏ (OOD)
        alpha=0.6,
        s=15,
        edgecolor=None,
        linewidth=0,
    )

    plt.title(
        "UMAP Visualization of Feature Space (EfficientNet-B3)",
        fontsize=15,
        fontweight="bold",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)

    legend = plt.legend(
        title="Dataset Domain",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        shadow=True,
    )
    # Cho legend nền trắng để không che điểm quá nhiều
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.9)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "UMAP_VISUALIZATION_FINAL.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🏁 Done! UMAP figure saved at: {save_path}")


if __name__ == "__main__":
    main()
