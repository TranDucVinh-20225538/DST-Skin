"""Frozen pathology foundation models for tile-level feature extraction.

UNI / Prov-GigaPath / Phikon. Encoders stay frozen. Transforms are the
published ImageNet-normalized 224² recipes — not MIDOG 50×50 stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Native patch size of the DST-Skin adapters (not the FM pretraining tile).
NATIVE_PATCH = {
    "camelyon17": 96,  # WILDS Camelyon17 patches
    "midog": 50,  # OpenMIBOOD mitosis crops
}


@dataclass(frozen=True)
class FMSpec:
    name: str
    repo: str
    backend: str  # timm | transformers | timm_hf_vit
    access_url: str
    gated: bool
    input_size: int
    expected_pretrain_tile: int
    feat_dim: int
    timm_id: str | None = None
    timm_kwargs: dict[str, Any] | None = None
    transformers_id: str | None = None
    native_notes: str = ""


SPECS: dict[str, FMSpec] = {
    "uni": FMSpec(
        name="uni",
        repo="MahmoodLab/UNI",
        backend="timm",
        access_url="https://huggingface.co/MahmoodLab/UNI",
        gated=True,
        input_size=224,
        expected_pretrain_tile=224,
        feat_dim=1024,
        timm_id="hf-hub:MahmoodLab/uni",
        timm_kwargs={
            "pretrained": True,
            "init_values": 1e-5,
            "dynamic_img_size": True,
            "num_classes": 0,
        },
        native_notes="ViT-L/16, Mass-100k WSI tiles at 224.",
    ),
    "gigapath": FMSpec(
        name="gigapath",
        repo="prov-gigapath/prov-gigapath",
        backend="timm",
        access_url="https://huggingface.co/prov-gigapath/prov-gigapath",
        gated=True,
        input_size=224,
        expected_pretrain_tile=256,
        feat_dim=1536,
        timm_id="hf_hub:prov-gigapath/prov-gigapath",
        timm_kwargs={"pretrained": True},
        native_notes="Tile encoder; official eval is Resize(256)+CenterCrop(224).",
    ),
    "phikon": FMSpec(
        name="phikon",
        repo="owkin/phikon",
        backend="timm_hf_vit",
        access_url="https://huggingface.co/owkin/phikon",
        gated=True,
        input_size=224,
        expected_pretrain_tile=224,
        feat_dim=768,
        transformers_id="owkin/phikon",
        native_notes=(
            "ViT-B/16 iBOT; CLS token. Not Phikon-v2. "
            "Loaded via timm + HF weights — torch-env transformers 5.x needs torch>=2.5."
        ),
    ),
}


def eval_transform(spec: FMSpec) -> transforms.Compose:
    size = spec.input_size
    if spec.name == "gigapath":
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def mismatch_record(domain: str, spec: FMSpec) -> dict[str, Any]:
    native = NATIVE_PATCH[domain]
    factor = spec.input_size / float(native)
    if domain == "midog":
        severity = "severe"
        note = (
            "MIDOG OpenMIBOOD crops are 50×50 mitosis patches; "
            f"{spec.repo} expects ~{spec.expected_pretrain_tile}² WSI tiles. "
            "A weak MIDOG result may be a resolution artifact, not an FM reversal."
        )
    else:
        severity = "moderate"
        note = (
            "WILDS Camelyon17 patches are 96×96 (typically 10x); "
            f"{spec.repo} expects ~{spec.expected_pretrain_tile}² tiles at 20x. "
            "Upsample is milder than MIDOG."
        )
    return {
        "domain": domain,
        "fm": spec.name,
        "native_patch_hw": native,
        "fm_input_hw": spec.input_size,
        "fm_pretrain_tile_hw": spec.expected_pretrain_tile,
        "upsample_factor": round(factor, 3),
        "mismatch_severity": severity,
        "note": note,
        "normalize": "imagenet",
        "midog_mean_std_used": False,
    }


class TimmFrozenEncoder(nn.Module):
    def __init__(self, spec: FMSpec):
        super().__init__()
        import timm

        kwargs = dict(spec.timm_kwargs or {})
        self.backbone = timm.create_model(spec.timm_id, **kwargs)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:
            out = out.flatten(start_dim=1)
        return out


class TransformersFrozenEncoder(nn.Module):
    def __init__(self, spec: FMSpec):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(spec.transformers_id)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        hidden = out.last_hidden_state
        return hidden[:, 0, :]


def _strip_hf_prefix(state: dict) -> dict:
    keys = list(state)
    for pref in ("vit.", "model.", "backbone."):
        n = sum(k.startswith(pref) for k in keys)
        if n > 0.5 * len(keys):
            return {
                (k[len(pref) :] if k.startswith(pref) else k): v
                for k, v in state.items()
            }
    return state


def hf_vit_state_to_timm(hf: dict) -> dict:
    """Map HuggingFace ViTModel keys onto a timm vit_base_patch16_224."""
    hf = _strip_hf_prefix(hf)
    out: dict = {}
    out["cls_token"] = hf["embeddings.cls_token"]
    out["pos_embed"] = hf["embeddings.position_embeddings"]
    out["patch_embed.proj.weight"] = hf["embeddings.patch_embeddings.projection.weight"]
    out["patch_embed.proj.bias"] = hf["embeddings.patch_embeddings.projection.bias"]
    n_layers = 12
    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        qw = hf[p + "attention.attention.query.weight"]
        kw = hf[p + "attention.attention.key.weight"]
        vw = hf[p + "attention.attention.value.weight"]
        qb = hf[p + "attention.attention.query.bias"]
        kb = hf[p + "attention.attention.key.bias"]
        vb = hf[p + "attention.attention.value.bias"]
        out[f"blocks.{i}.attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
        out[f"blocks.{i}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
        out[f"blocks.{i}.attn.proj.weight"] = hf[p + "attention.output.dense.weight"]
        out[f"blocks.{i}.attn.proj.bias"] = hf[p + "attention.output.dense.bias"]
        out[f"blocks.{i}.norm1.weight"] = hf[p + "layernorm_before.weight"]
        out[f"blocks.{i}.norm1.bias"] = hf[p + "layernorm_before.bias"]
        out[f"blocks.{i}.norm2.weight"] = hf[p + "layernorm_after.weight"]
        out[f"blocks.{i}.norm2.bias"] = hf[p + "layernorm_after.bias"]
        out[f"blocks.{i}.mlp.fc1.weight"] = hf[p + "intermediate.dense.weight"]
        out[f"blocks.{i}.mlp.fc1.bias"] = hf[p + "intermediate.dense.bias"]
        out[f"blocks.{i}.mlp.fc2.weight"] = hf[p + "output.dense.weight"]
        out[f"blocks.{i}.mlp.fc2.bias"] = hf[p + "output.dense.bias"]
    out["norm.weight"] = hf["layernorm.weight"]
    out["norm.bias"] = hf["layernorm.bias"]
    return out


def _load_hf_vit_weights(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(repo, filename=fname)
        except Exception as exc:  # gated / missing filename
            last_err = exc
            continue
        if fname.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path)
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        return payload
    raise RuntimeError(f"Could not download ViT weights for {repo}: {last_err}")


class TimmHfVitFrozenEncoder(nn.Module):
    """ViT-B/16 CLS encoder from a HuggingFace ViTModel checkpoint, no transformers import."""

    def __init__(self, spec: FMSpec):
        super().__init__()
        import timm

        repo = spec.transformers_id or spec.repo
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )
        converted = hf_vit_state_to_timm(_load_hf_vit_weights(repo))
        missing, unexpected = self.backbone.load_state_dict(converted, strict=False)
        missing = [k for k in missing if not k.startswith("head.")]
        if missing:
            raise RuntimeError(f"Phikon/timm missing keys: {missing}")
        if unexpected:
            print(f"timm_hf_vit unexpected keys (ignored): {unexpected}", flush=True)
        print(f"Loaded {repo} via timm ViT-B/16 CLS (no transformers)", flush=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:
            out = out.flatten(start_dim=1)
        return out


def load_frozen_encoder(name: str) -> tuple[nn.Module, FMSpec]:
    if name not in SPECS:
        raise ValueError(f"Unknown FM {name}; choose from {list(SPECS)}")
    spec = SPECS[name]
    if spec.backend == "timm":
        model = TimmFrozenEncoder(spec)
    elif spec.backend == "transformers":
        model = TransformersFrozenEncoder(spec)
    elif spec.backend == "timm_hf_vit":
        model = TimmHfVitFrozenEncoder(spec)
    else:
        raise ValueError(spec.backend)
    model.eval()
    return model, spec
