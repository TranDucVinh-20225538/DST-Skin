"""WILDS iWildCam loaders — location covariate shift (non-medical #5)."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from wilds import get_dataset

from src.models.cnn_family import input_size as cnn_input_size

IWILDCAM_NUM_CLASSES = 186
DATA_ROOT = "data/raw/wilds"
OOD_SPLIT = "test"  # official OOD/trans locations


class _WildsPathDataset(Dataset):
    def __init__(self, subset, prefix: str):
        self.subset = subset
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label, _metadata = self.subset[idx]
        return image, int(label), f"{self.prefix}/{idx}"


def data_ready(root: str = DATA_ROOT) -> Path:
    return Path(root) / "iwildcam_v2.0" / "metadata.csv"


def get_wilds_dataset(download: bool = False):
    return get_dataset(
        dataset="iwildcam",
        root_dir=DATA_ROOT,
        download=download,
        split_scheme="official",
    )


def build_transform(backbone: str, train: bool = False):
    size = cnn_input_size(backbone)
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize(int(round(size * 256 / 224))),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(256 if size == 224 else size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    backbone: str,
    batch_size: int = 64,
    num_workers: int = 4,
    include_ood: bool = True,
    download: bool = False,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    dataset = get_wilds_dataset(download=download)
    train_tf = build_transform(backbone, train=True)
    eval_tf = build_transform(backbone, train=False)
    train_subset = dataset.get_subset("train", transform=train_tf)
    id_val_subset = dataset.get_subset("id_val", transform=eval_tf)
    loaders = {
        "train": DataLoader(
            _WildsPathDataset(train_subset, "iwildcam/train"),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "id_val": DataLoader(
            _WildsPathDataset(id_val_subset, "iwildcam/id_val"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    if include_ood:
        ood_subset = dataset.get_subset(OOD_SPLIT, transform=eval_tf)
        loaders["ood"] = DataLoader(
            _WildsPathDataset(ood_subset, f"iwildcam/{OOD_SPLIT}"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def split_summary(download: bool = False) -> dict:
    dataset = get_wilds_dataset(download=download)
    summary = {
        "ood_split": OOD_SPLIT,
        "n_classes": int(dataset.n_classes),
    }
    for split in ("train", "id_val", "val", "test", "id_test"):
        if split in dataset.split_dict:
            n = int((dataset.split_array == dataset.split_dict[split]).sum())
            summary[split] = n
    return summary
