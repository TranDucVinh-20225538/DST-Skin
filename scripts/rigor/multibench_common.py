"""Shared pieces of the group-leakage multibench check (decisions/precommit_group_leakage_multibench_2026-10-06.md).

Dataset specs, the locked group folds (default_rng(1042), greedy by train-image count within strata),
loaders that keep the WILDS split order (index-aligned with metadata), and model checkpoints.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FOLD_RNG = 1042
ARCHS_B = ("resnet18", "resnet50", "densenet121", "convnext_tiny")
ARCHS_A = {
    "iwildcam": ("resnet18", "resnet50", "densenet121", "convnext_tiny", "mobilenet_v3_large",
                 "regnet_y_3_2gf", "effb3", "efficientnet_v2_s"),
    "rxrx1": ARCHS_B,
}
SPECS = {
    "iwildcam": {"wilds": "iwildcam", "dir": "iwildcam_v2.0", "id": "id_val", "ood": "test", "group": "location",
                 "stratum": None, "n_classes": 186, "select": "id_val_best"},
    "rxrx1": {"wilds": "rxrx1", "dir": "rxrx1_v1.0", "id": "id_test", "ood": "test", "group": "experiment",
              "stratum": "cell_type", "n_classes": 1139, "select": "last"},
}


def wilds_root() -> str:
    return os.environ.get("DST_WILDS_ROOT", str(C.REPO / "data/raw/wilds"))


def get_dataset(ds: str):
    from wilds import get_dataset as gd
    return gd(dataset=SPECS[ds]["wilds"], root_dir=wilds_root(), download=False)


def split_idx(dset, split: str) -> np.ndarray:
    return np.where(dset.split_array == dset.split_dict[split])[0]


def meta_col(dset, name: str) -> np.ndarray:
    return np.asarray(dset.metadata_array[:, dset.metadata_fields.index(name)]).astype(np.int64)


def folds(dset, ds: str):
    """Fold of every train / ID / OOD index (-1 = group without train images). Locked in the precommit."""
    sp = SPECS[ds]
    g = meta_col(dset, sp["group"])
    st = meta_col(dset, sp["stratum"]) if sp["stratum"] else np.zeros(len(g), dtype=np.int64)
    tr = split_idx(dset, "train")
    rng = np.random.default_rng(FOLD_RNG)
    cnt = dict(zip(*np.unique(g[tr], return_counts=True)))
    fold_of, load = {}, [0, 0]
    for s in sorted(np.unique(st[tr])):
        groups = np.array(sorted(np.unique(g[tr][st[tr] == s])))
        rng.shuffle(groups)
        for x in groups:
            f = 0 if load[0] <= load[1] else 1
            fold_of[int(x)] = f
            load[f] += int(cnt[x])
    return np.array([fold_of.get(int(x), -1) for x in g]), g


def transform(ds: str, arch: str, train: bool):
    from torchvision import transforms as T
    if ds == "iwildcam":
        from src.datasets.iwildcam_ood import build_transform
        return build_transform(arch, train=train)
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        rot = T.Lambda(lambda im: im.rotate(90 * random.randint(0, 3)))
        return T.Compose([T.Resize(224), T.RandomHorizontalFlip(), rot, T.ToTensor(), norm])
    return T.Compose([T.Resize(224), T.ToTensor(), norm])


def loader(dset, idx, tfm, bs, shuffle, workers=8):
    from torch.utils.data import DataLoader, Dataset
    from wilds.datasets.wilds_dataset import WILDSSubset

    class _Tup(Dataset):
        def __init__(self, sub):
            self.sub = sub

        def __len__(self):
            return len(self.sub)

        def __getitem__(self, i):
            x, y, _ = self.sub[i]
            return x, int(y), i

    return DataLoader(_Tup(WILDSSubset(dset, idx, tfm)), batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=True)


def published_ckpt(ds: str, arch: str) -> Path:
    if ds == "iwildcam":
        return C.REPO / "data/models/iwildcam/seed42" / ("%s_best.pth" % arch)
    return C.REPO / "data/models/multibench/rxrx1/seed42" / ("%s_full_last.pth" % arch)


def feat_dir(ds: str) -> Path:
    return C.ensure_dir(C.default_out(C.REPO) / "multibench" / ds)


def report_dir(ds: str | None = None) -> Path:
    d = C.default_reports(C.REPO) / "multibench"
    return C.ensure_dir(d / ds if ds else d)
