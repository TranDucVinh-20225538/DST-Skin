"""CIFAR-10-C (Hendrycks & Dietterich 2019) loaders for controlled severity sweep.

ID = clean CIFAR-10 test (same order as the .npy stacks). OOD = one (corruption, severity).
Yields (image, label, path) like camelyon_ood / midog_ood.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.datasets.cifar_ood import DATA_ROOT, _PathDataset, build_transform
from torchvision import datasets

CIFAR10C_ROOT = Path(f"{DATA_ROOT}/cifar10c")
N_PER_SEVERITY = 10_000
N_SEVERITY = 5

# Official CIFAR-10-C 15 corruptions (Hendrycks & Dietterich 2019).
CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


class CIFAR10C(Dataset):
    """One corruption type at one severity (1–5). Aligned 1:1 with CIFAR-10 test."""

    def __init__(
        self,
        corruption_type: str,
        severity: int,
        root: str | Path = CIFAR10C_ROOT,
        transform=None,
    ):
        if corruption_type not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption {corruption_type}")
        if severity not in range(1, N_SEVERITY + 1):
            raise ValueError("severity must be 1..5")
        root = Path(root)
        images = np.load(root / f"{corruption_type}.npy")
        labels = np.load(root / "labels.npy")
        start = (severity - 1) * N_PER_SEVERITY
        stop = start + N_PER_SEVERITY
        self.images = images[start:stop]
        self.labels = labels[start:stop].astype(np.int64)
        if len(self.images) != N_PER_SEVERITY:
            raise RuntimeError(
                f"{corruption_type} severity {severity}: expected "
                f"{N_PER_SEVERITY} images, got {len(self.images)}"
            )
        self.transform = transform
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        path = f"cifar10c/{self.corruption_type}/s{self.severity}/{idx}"
        return image, label, path


def data_ready(root: str | Path = CIFAR10C_ROOT) -> Path:
    return Path(root) / "labels.npy"


def get_cifar10c_loader(
    backbone: str,
    corruption_type: str,
    severity: int,
    batch_size: int = 256,
    num_workers: int = 4,
    root: str | Path = CIFAR10C_ROOT,
) -> DataLoader:
    tf = build_transform(backbone, train=False)
    ds = CIFAR10C(corruption_type, severity, root=root, transform=tf)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_clean_cifar10_test_loader(
    backbone: str,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    tf = build_transform(backbone, train=False)
    cifar_root = f"{DATA_ROOT}/cifar10"
    test = datasets.CIFAR10(root=cifar_root, train=False, download=False, transform=tf)
    return DataLoader(
        _PathDataset(test, "cifar10/test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
