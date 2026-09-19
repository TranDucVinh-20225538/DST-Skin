"""CIFAR-10 (ID) and SVHN (OOD) loaders for cross-domain pilot."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR10_NUM_CLASSES = 10
DATA_ROOT = "data/raw"


class _PathDataset(Dataset):
    def __init__(self, base: Dataset, prefix: str):
        self.base = base
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, f"{self.prefix}/{idx}"


def _input_size(backbone: str) -> int:
    return 300 if backbone == "effb3" else 224


def build_transform(backbone: str, train: bool = False) -> transforms.Compose:
    size = _input_size(backbone)
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    if train:
        resize = 320 if size == 300 else 256
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    resize = size if size == 300 else 256
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(
    backbone: str,
    batch_size: int = 128,
    num_workers: int = 4,
    include_ood: bool = True,
) -> dict[str, DataLoader]:
    """Load CIFAR ID splits; SVHN (OOD) optional so train is not blocked on download."""
    train_tf = build_transform(backbone, train=True)
    eval_tf = build_transform(backbone, train=False)

    cifar_root = f"{DATA_ROOT}/cifar10"
    svhn_root = f"{DATA_ROOT}/svhn"

    cifar_train = datasets.CIFAR10(
        root=cifar_root, train=True, download=True, transform=train_tf
    )
    cifar_test = datasets.CIFAR10(
        root=cifar_root, train=False, download=True, transform=eval_tf
    )

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            _PathDataset(cifar_train, "cifar10/train"),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "id_val": DataLoader(
            _PathDataset(cifar_test, "cifar10/test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if include_ood:
        svhn_test = datasets.SVHN(
            root=svhn_root, split="test", download=True, transform=eval_tf
        )
        loaders["ood"] = DataLoader(
            _PathDataset(svhn_test, "svhn/test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
