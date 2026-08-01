# scripts/extract_once_efficientnet_b3.py
import os
import torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from src.datasets.isic_dataset import ISICDataset
from src.models.resnet18 import get_resnet18
from src.utils.feature_extractor import extract_features_and_logits

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

def build_transform():
    return transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def get_loader(csv_path, img_root, batch_size=32):
    ds = ISICDataset(csv_path, img_root, transform=build_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, len(ds)

def main():
    os.makedirs("outputs/features", exist_ok=True)

    print(f" Extracting features/logits with Resnet18 on {DEVICE}")

    # 1. Model
    model = get_resnet18(num_classes=2, pretrained=False)
    state = torch.load("data/models/resnet18_robust_best.pth",
                       map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    # 2. Loaders
    train_csv = "data/processed/isic2018_binary/isic_2018_binary_train.csv"
    train_root = "data/raw/isic2018/ISIC2018_Task3_Training_Input"

    val_csv = "data/processed/isic2018_binary/isic_2018_binary_val.csv"
    val_root = "data/raw/isic2018/ISIC2018_Task3_Validation_Input"

    ood_csv = "data/processed/pad_ufes20_binary/pad_ufes_binary.csv"
    ood_root = "data/raw/pad_ufes20/images"

    train_loader, _ = get_loader(train_csv, train_root)
    val_loader, _   = get_loader(val_csv, val_root)
    ood_loader, _   = get_loader(ood_csv, ood_root)

    # 3. Extract
    print("--- ISIC Train ---")
    train_logits, train_feats = extract_features_and_logits(
        model, train_loader, DEVICE, target_layer=None
    )

    print("--- ISIC Val ---")
    val_logits, val_feats = extract_features_and_logits(
        model, val_loader, DEVICE, target_layer=None
    )

    print("--- PAD-UFES (OOD) ---")
    ood_logits, ood_feats = extract_features_and_logits(
        model, ood_loader, DEVICE, target_layer=None
    )

    # 4. Save tensors + labels (labels lấy trực tiếp từ dataset CSV)
    def load_labels(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df["binary_label"].values.astype(np.int64)

    train_labels = load_labels(train_csv)
    val_labels   = load_labels(val_csv)
    ood_labels   = load_labels(ood_csv)
    fc_weight = model.fc.weight.detach().cpu()
    fc_bias   = model.fc.bias.detach().cpu()

    save_path = "outputs/features/resnet18_isic_pad_features.pt"
    torch.save({
        "train_logits": train_logits,
        "train_feats": train_feats,
        "train_labels": train_labels,
        "val_logits": val_logits,
        "val_feats": val_feats,
        "val_labels": val_labels,
        "ood_logits": ood_logits,
        "ood_feats": ood_feats,
        "ood_labels": ood_labels,
        "fc_weight": fc_weight,
        "fc_bias": fc_bias,
    }, save_path)

    print(f"Saved all features/logits to {save_path}")

if __name__ == "__main__":
    main()