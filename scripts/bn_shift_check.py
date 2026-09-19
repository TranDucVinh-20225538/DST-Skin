#!/usr/bin/env python3
"""BN activation-stat shift (Schneider et al. 2020) as one mechanism check.

ID vs OOD channel-wise mean shift at BatchNorm inputs, per (domain, backbone).
No retrain. CPU. Does not touch CIFAR-10-C or locked metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.datasets.camelyon_ood import get_dataloaders as get_camelyon_loaders
from src.datasets.isic_dataset import ISICDataset
from src.datasets.midog_ood import (
    CKPT_PATH,
    DATA_ROOT as MIDOG_ROOT,
    MIDOG_NUM_CLASSES,
    get_dataloaders as get_midog_loaders,
)
from src.models.efficientnet_b3 import get_efficientnet_b3
from src.models.resnet18 import get_resnet18
from src.models.resnet50 import get_resnet50

GAP = Path("outputs/reports/fid_vs_gap_covariate.csv")
OUT = Path("outputs/reports/bn_shift_check.csv")
SEED = 42
MAX_PER_SPLIT = 512
BATCH = 16
EPS = 1e-6


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cap(dataset, n: int, batch: int) -> DataLoader:
    n = min(n, len(dataset))
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(dataset), size=n, replace=False))
    return DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=batch,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )


class BNStatHook:
    def __init__(self, model: nn.Module):
        self.layers: list[str] = []
        self._sum: dict[str, torch.Tensor] = {}
        self._n: dict[str, float] = {}
        self.handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                self.layers.append(name)
                self.handles.append(mod.register_forward_hook(self._make(name)))

    def _make(self, name: str):
        def hook(_mod, inp, _out):
            x = inp[0].detach()
            if x.dim() == 4:
                reduce = (0, 2, 3)
                n = x.shape[0] * x.shape[2] * x.shape[3]
            elif x.dim() == 2:
                reduce = (0,)
                n = x.shape[0]
            else:
                return
            s = x.sum(dim=reduce)
            if name not in self._sum:
                self._sum[name] = torch.zeros_like(s)
                self._n[name] = 0.0
            self._sum[name] += s.cpu()
            self._n[name] += float(n)

        return hook

    def means(self) -> dict[str, np.ndarray]:
        return {k: (self._sum[k] / self._n[k]).numpy() for k in self.layers if k in self._sum}

    def reset(self) -> None:
        self._sum.clear()
        self._n.clear()

    def close(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_split(model: nn.Module, loader: DataLoader, hook: BNStatHook, device: torch.device) -> dict[str, np.ndarray]:
    hook.reset()
    model.eval()
    for batch in tqdm(loader, desc="BN", leave=False):
        images = batch[0].to(device)
        model(images)
    return hook.means()


def bn_shift(mu_id: dict[str, np.ndarray], mu_ood: dict[str, np.ndarray]) -> float:
    """Mean over layers of RMS channel shift, standardized by ID scale (per-layer RMS of μ_id)."""
    vals = []
    for name in mu_id:
        if name not in mu_ood:
            continue
        a, b = mu_id[name], mu_ood[name]
        scale = float(np.sqrt(np.mean(a * a) + EPS))
        vals.append(float(np.sqrt(np.mean((b - a) ** 2)) / scale))
    return float(np.mean(vals)) if vals else float("nan")


def get_backbone(name: str, num_classes: int) -> nn.Module:
    if name == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=False)
    if name == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=False)
    if name == "effb3":
        return get_efficientnet_b3(num_classes=num_classes, pretrained=False)
    raise ValueError(name)


def load_state(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def skin_loaders(backbone: str):
    # Match extract_once_*: 300×300 ImageNet-norm for all skin backbones.
    tf = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    id_ds = ISICDataset(
        "data/processed/isic2018_binary/isic_2018_binary_val.csv",
        "data/raw/isic2018/ISIC2018_Task3_Validation_Input",
        transform=tf,
    )
    ood_ds = ISICDataset(
        "data/processed/pad_ufes20_binary/pad_ufes_binary.csv",
        "data/raw/pad_ufes20/images",
        transform=tf,
    )
    return cap(id_ds, MAX_PER_SPLIT, BATCH), cap(ood_ds, MAX_PER_SPLIT, BATCH)


def camelyon_loaders(backbone: str):
    loaders = get_camelyon_loaders(
        backbone,
        batch_size=BATCH,
        num_workers=2,
        include_ood=True,
        download=False,
        train_frac=0.05,
        seed=SEED,
    )
    return cap(loaders["id_val"].dataset, MAX_PER_SPLIT, BATCH), cap(
        loaders["ood"].dataset, MAX_PER_SPLIT, BATCH
    )


def midog_loaders(backbone: str):
    dl_bb = None if backbone == "resnet50" else backbone
    loaders = get_midog_loaders(
        batch_size=BATCH,
        num_workers=2,
        include_ood=True,
        data_root=MIDOG_ROOT,
        backbone=dl_bb,
    )
    return cap(loaders["id_test"].dataset, MAX_PER_SPLIT, BATCH), cap(
        loaders["ood"].dataset, MAX_PER_SPLIT, BATCH
    )


def midog_r50(device: torch.device) -> nn.Module:
    model = get_resnet50(num_classes=MIDOG_NUM_CLASSES, pretrained=False)
    try:
        state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    device = torch.device("cpu")
    gap = pd.read_csv(GAP)
    jobs = []
    for bb in ("resnet18", "resnet50", "effb3"):
        ckpt = Path(f"data/models/{bb if bb != 'effb3' else 'efficientnet_b3'}_robust_best.pth")
        jobs.append(("skin_isic_pad", bb, 2, ckpt, skin_loaders))
        jobs.append(
            (
                "camelyon17",
                bb,
                2,
                Path(f"data/models/camelyon17/frac1/seed42/{bb}_best.pth"),
                camelyon_loaders,
            )
        )
        if bb == "resnet50":
            jobs.append(("midog", bb, MIDOG_NUM_CLASSES, Path("OPENMIBOOD"), midog_loaders))
        else:
            jobs.append(
                (
                    "midog",
                    bb,
                    MIDOG_NUM_CLASSES,
                    Path(f"data/models/midog/seed42/{bb}_best.pth"),
                    midog_loaders,
                )
            )

    rows = []
    for domain, backbone, ncls, ckpt, factory in jobs:
        print(f"\n=== BN {domain}/{backbone} ===", flush=True)
        if domain == "midog" and backbone == "resnet50":
            model = midog_r50(device)
        else:
            model = load_state(get_backbone(backbone, ncls), ckpt, device)
        hook = BNStatHook(model)
        print(f"  {len(hook.layers)} BN layers", flush=True)
        id_loader, ood_loader = factory(backbone)
        mu_id = run_split(model, id_loader, hook, device)
        mu_ood = run_split(model, ood_loader, hook, device)
        shift = bn_shift(mu_id, mu_ood)
        hook.close()
        g = gap[(gap["domain"] == domain) & (gap["backbone"] == backbone)].iloc[0]
        rows.append({
            "domain": domain,
            "backbone": backbone,
            "bn_shift": shift,
            "n_bn_layers": len(hook.layers),
            "logit_gap_trim1": float(g["logit_gap_mean_gap_vs_id"]),
            "delta_auroc": float(g["auroc_delta_maha_minus_msp"]),
        })
        print(f"  bn_shift={shift:.4f}", flush=True)
        del model

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(df.to_string(index=False))
    rho_g, p_g = spearmanr(df["bn_shift"], df["logit_gap_trim1"].abs())
    rho_d, p_d = spearmanr(df["bn_shift"], df["delta_auroc"])
    print(f"\nPooled Spearman n={len(df)}:")
    print(f"  BN-shift vs |LogitGap|: ρ={rho_g:.3f} p={p_g:.3f}")
    print(f"  BN-shift vs ΔAUROC:     ρ={rho_d:.3f} p={p_d:.3f}")
    cam = df[df["domain"] == "camelyon17"].set_index("backbone")
    print("\nCamelyon ranking (hyp: larger BN-shift → EffB3-style high |LogitGap|, low Δ):")
    for bb in ("effb3", "resnet50", "resnet18"):
        r = cam.loc[bb]
        print(
            f"  {bb:8s}  BN={r['bn_shift']:.4f}  |LogitGap|={abs(r['logit_gap_trim1']):.3f}  "
            f"Δ={r['delta_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
