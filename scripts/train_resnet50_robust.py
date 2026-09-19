import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from src.datasets.isic_dataset import ISICDataset
from src.models.resnet50 import get_resnet50


def build_train_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.GaussianBlur(kernel_size=(3, 5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_dataloaders(batch_size=64, num_workers=0):
    train_tf = build_train_transform()
    val_tf   = build_val_transform()

    train_ds = ISICDataset(
        "data/processed/isic2018_binary/isic_2018_binary_train.csv",
        "data/raw/isic2018/ISIC2018_Task3_Training_Input",
        transform=train_tf
    )
    val_ds = ISICDataset(
        "data/processed/isic2018_binary/isic_2018_binary_val.csv",
        "data/raw/isic2018/ISIC2018_Task3_Validation_Input",
        transform=val_tf
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # lớp ác tính

    # Accuracy (chỉ để tham khảo)
    preds = (probs >= 0.5).astype(int)
    acc = (preds == labels).mean()

    # AUROC – thước đo để chọn best model
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0  # phòng khi chỉ có 1 lớp (không xảy ra với split chuẩn)

    return acc, auc


def train_resnet50_robust():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

    os.makedirs("data/models", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"🚀 Training ResNet-50 CE-Robust on {device}")

    # Bạn có thể chỉnh num_workers tùy Mac / GPU
    train_loader, val_loader = get_dataloaders(batch_size=64, num_workers=0)

    model = get_resnet50(num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    best_val_auc = 0.0
    best_path = "data/models/resnet50_robust_best.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        val_acc, val_auc = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]

        print(f"[Epoch {epoch:02d}/{num_epochs}] "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}")

        # Lưu best model theo Val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model to {best_path} (Val AUC {best_val_auc:.4f})")

    # Lưu thêm checkpoint cuối cùng nếu muốn
    final_path = "data/models/resnet50_robust_epoch_last.pth"
    torch.save(model.state_dict(), final_path)
    print(f"💾 Saved final model to {final_path}")


if __name__ == "__main__":
    train_resnet50_robust()
