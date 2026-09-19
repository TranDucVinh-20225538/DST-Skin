"""WILDS Camelyon17 loaders for covariate-shift OOD pilot (binary tumor vs normal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset

CAMELYON_NUM_CLASSES = 2
DATA_ROOT = "data/raw/wilds"
OOD_SPLIT = "test"  # hospital 2 — canonical WILDS OOD test center


class _WildsPathDataset(Dataset):
    def __init__(self, subset, prefix: str):
        self.subset = subset
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label, _metadata = self.subset[idx]
        return image, label, f"{self.prefix}/{idx}"


def _input_size(backbone: str) -> int:
    from src.models.cnn_family import input_size

    return input_size(backbone)


def build_transform(
    backbone: str,
    train: bool = False,
    input_size: int | None = None,
    stain_cov: bool = False,
) -> transforms.Compose:
    """Upscale native 96×96 WILDS patches to each backbone's ImageNet input size.

    `stain_cov`: HED jitter on the **train** transform only (Viên 1).
    Eval / OOD stays ImageNet-normalize, no stain jitter.
    """
    size = int(input_size) if input_size is not None else _input_size(backbone)
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    if train:
        resize = int(round(size * 256 / 224))
        ops: list = [
            transforms.Resize(resize),
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if stain_cov:
            from src.datasets.stain_aug import HEDJitter, STAIN_COV_P, STAIN_COV_SIGMA

            ops.append(HEDJitter(sigma=STAIN_COV_SIGMA, p=STAIN_COV_P))
        ops.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(ops)
    resize = 256 if size == 224 else size
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


def data_ready(root: str = DATA_ROOT) -> Path:
    """Return release marker path when Camelyon17 is fully extracted."""
    marker = Path(root) / "camelyon17_v1.0" / "RELEASE_v1.0.txt"
    return marker


def get_wilds_dataset(download: bool = False):
    return get_dataset(
        dataset="camelyon17",
        root_dir=DATA_ROOT,
        download=download,
        split_scheme="official",
    )


def stratified_train_indices(dataset, frac: float, seed: int) -> np.ndarray:
    """Subsample train split stratified by (hospital, tumor label), not WILDS random frac."""
    train_split = dataset.split_dict["train"]
    split_idx = np.where(dataset.split_array == train_split)[0]
    if frac >= 1.0:
        return split_idx

    meta = dataset.metadata_array[split_idx].numpy()
    hospitals = meta[:, 0]
    labels = meta[:, 2]
    rng = np.random.default_rng(seed)

    chunks: list[np.ndarray] = []
    for hospital in sorted(np.unique(hospitals)):
        for label in (0, 1):
            mask = (hospitals == hospital) & (labels == label)
            group_idx = split_idx[mask]
            if len(group_idx) == 0:
                continue
            n_retain = max(1, int(round(len(group_idx) * frac)))
            n_retain = min(n_retain, len(group_idx))
            chunks.append(rng.choice(group_idx, size=n_retain, replace=False))

    return np.sort(np.concatenate(chunks))


def log_train_subsample(dataset, indices: np.ndarray, frac: float, seed: int) -> None:
    """Print hospital × label counts for reproducibility checks."""
    if frac >= 1.0:
        return
    meta = dataset.metadata_array[indices].numpy()
    print(
        f"Train subsample: frac={frac}, seed={seed}, n={len(indices)} "
        f"(full train={int((dataset.split_array == dataset.split_dict['train']).sum())})"
    )
    for hospital in sorted(np.unique(meta[:, 0])):
        for label in (0, 1):
            n = int(((meta[:, 0] == hospital) & (meta[:, 2] == label)).sum())
            if n:
                print(f"  hospital={int(hospital)} label={int(label)}: {n}")


def get_dataloaders(
    backbone: str,
    batch_size: int = 128,
    num_workers: int = 4,
    include_ood: bool = True,
    download: bool = False,
    train_frac: float = 1.0,
    seed: int = 42,
    transform=None,
    shuffle_train: bool = True,
    input_size: int | None = None,
    stain_cov: bool = False,
    ood_split: str | None = None,
) -> dict[str, DataLoader]:
    """WILDS official split: train/id_val from hospitals 0,3,4; OOD = hospital 2 (test).

    `transform`, if given, is used for every split (frozen-encoder extract: no train aug).
    `stain_cov` only affects the train transform. `ood_split` defaults to hospital-2 test;
    pass ``val`` for hospital 1.
    """
    dataset = get_wilds_dataset(download=download)
    if transform is not None:
        train_tf = eval_tf = transform
    else:
        train_tf = build_transform(
            backbone, train=True, input_size=input_size, stain_cov=stain_cov
        )
        eval_tf = build_transform(
            backbone, train=False, input_size=input_size, stain_cov=False
        )

    train_idx = stratified_train_indices(dataset, train_frac, seed)
    log_train_subsample(dataset, train_idx, train_frac, seed)
    train_subset = WILDSSubset(dataset, train_idx, train_tf)
    id_val_subset = dataset.get_subset("id_val", transform=eval_tf)

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            _WildsPathDataset(train_subset, "camelyon17/train"),
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "id_val": DataLoader(
            _WildsPathDataset(id_val_subset, "camelyon17/id_val"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if include_ood:
        split = ood_split or OOD_SPLIT
        ood_subset = dataset.get_subset(split, transform=eval_tf)
        loaders["ood"] = DataLoader(
            _WildsPathDataset(ood_subset, f"camelyon17/{split}"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def split_summary(download: bool = False) -> dict[str, int | str]:
    """Hospital-aware split counts for Bước 0 reporting."""
    dataset = get_wilds_dataset(download=download)
    summary: dict[str, int | str] = {"ood_split": OOD_SPLIT}
    for split in ("train", "id_val", "val", "test"):
        if split in dataset.split_dict:
            n = int((dataset.split_array == dataset.split_dict[split]).sum())
            summary[split] = n
    centers = dataset.metadata_array[:, 0].numpy()
    for split in ("train", "id_val", "val", "test"):
        if split not in dataset.split_dict:
            continue
        mask = dataset.split_array == dataset.split_dict[split]
        hospitals = sorted(set(int(c) for c in centers[mask]))
        summary[f"{split}_hospitals"] = ",".join(map(str, hospitals))
    return summary
