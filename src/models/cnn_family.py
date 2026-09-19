"""CNN-family backbones for architecture-invariance expansion.

All expose an `avgpool` module so feature_extractor.py's default hook works
without changes. DenseNet121 is wrapped to add that module (torchvision's
DenseNet pools via F.adaptive_avg_pool2d, not a named layer).
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Original three stay named as in existing pilots.
ORIGINAL_BACKBONES = ("resnet18", "resnet50", "effb3")
NEW_CNN_BACKBONES = (
    "densenet121",
    "convnext_tiny",
    "mobilenet_v3_large",
    "regnet_y_3_2gf",
    "efficientnet_v2_s",
)
# 9th backbone (#4). Not in NEW_CNN; rebuild must skip so Camelyon-8 W stays 8 CNNs.
VIT_BACKBONES = ("vit_b_16",)

# Standard fine-tune recipes (not forced-identical to ResNet Adam 1e-4).
# Epochs / data / seed stay in the caller.
RECIPES: dict[str, dict[str, Any]] = {
    "resnet18": {"optim": "adam", "lr": 1e-4, "wd": 1e-4, "batch_size": 64},
    "resnet50": {"optim": "adam", "lr": 1e-4, "wd": 1e-4, "batch_size": 64},
    "effb3": {"optim": "adam", "lr": 1e-4, "wd": 1e-4, "batch_size": 64},
    "densenet121": {"optim": "adam", "lr": 1e-4, "wd": 1e-4, "batch_size": 64},
    "convnext_tiny": {"optim": "adamw", "lr": 1e-4, "wd": 0.05, "batch_size": 64},
    "mobilenet_v3_large": {"optim": "adamw", "lr": 1e-4, "wd": 0.01, "batch_size": 64},
    "regnet_y_3_2gf": {
        "optim": "sgd",
        "lr": 5e-3,
        "wd": 5e-5,
        "momentum": 0.9,
        "batch_size": 64,
    },
    "efficientnet_v2_s": {"optim": "adamw", "lr": 1e-4, "wd": 1e-5, "batch_size": 32},
    "vit_b_16": {"optim": "adamw", "lr": 1e-4, "wd": 0.05, "batch_size": 32},
}

# Family tags for A/B/C: is the Camelyon MSP jump EffB3-only or cross-family?
ARCH_FAMILY = {
    "resnet18": "resnet_like",
    "resnet50": "resnet_like",
    "effb3": "efficientnet_like",
    "efficientnet_v2_s": "efficientnet_like",
    "densenet121": "dense_connection",
    "convnext_tiny": "modern_cnn",
    "mobilenet_v3_large": "lightweight",
    "regnet_y_3_2gf": "nas_designed",
    "vit_b_16": "transformer",
}
ARCH_FAMILY_LABEL = {
    "resnet_like": "ResNet-like (R18/R50)",
    "efficientnet_like": "compound-scaled / EfficientNet-like (EffB3, EffV2-S)",
    "dense_connection": "dense-connection (DenseNet121)",
    "modern_cnn": "modern-CNN (ConvNeXt-Tiny)",
    "lightweight": "lightweight (MobileNetV3-Large)",
    "nas_designed": "NAS-designed (RegNetY-3.2GF)",
    "transformer": "ViT-B/16 (held-out family)",
}

_INPUT_SIZE = {
    "effb3": 300,
    "efficientnet_v2_s": 384,
}


def input_size(backbone: str) -> int:
    return int(_INPUT_SIZE.get(backbone, 224))


def recipe(backbone: str) -> dict[str, Any]:
    if backbone not in RECIPES:
        raise ValueError(f"No training recipe for {backbone}")
    return dict(RECIPES[backbone])


def last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            last = mod
    if last is None:
        raise ValueError("No nn.Linear head found")
    return last


def fc_params(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    lin = last_linear(model)
    return lin.weight.detach().cpu(), lin.bias.detach().cpu()


def _densenet121(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    model.forward = types.MethodType(_forward, model)
    return model


def _convnext_tiny(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


def _mobilenet_v3_large(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    # Stock head is Linear(960,1280)→Hardswish→Dropout→Linear(1280,C).
    # feature_extractor hooks avgpool (960-d); ViM/ReAct need W.shape[1]==feat_dim,
    # so replace with a single linear on avgpool (ResNet-style penultimate).
    in_f = model.classifier[0].in_features
    model.classifier = nn.Linear(in_f, num_classes)
    return model


def _regnet_y_3_2gf(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.regnet_y_3_2gf(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _efficientnet_v2_s(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model


class _CLSPool(nn.Module):
    """Hook target for ViT: encoder output [B, seq, dim] → CLS [B, dim]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x[:, 0]
        return x


def _vit_b_16(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    in_f = model.heads.head.in_features
    model.heads.head = nn.Linear(in_f, num_classes)
    model.avgpool = _CLSPool()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._process_input(x)
        n = x.shape[0]
        x = torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)
        x = self.encoder(x)
        x = self.avgpool(x)
        return self.heads(x)

    model.forward = types.MethodType(_forward, model)
    return model


_NEW_BUILDERS = {
    "densenet121": _densenet121,
    "convnext_tiny": _convnext_tiny,
    "mobilenet_v3_large": _mobilenet_v3_large,
    "regnet_y_3_2gf": _regnet_y_3_2gf,
    "efficientnet_v2_s": _efficientnet_v2_s,
    "vit_b_16": _vit_b_16,
}


def get_cnn_backbone(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Original three plus the five new CNN-family models."""
    from src.models.efficientnet_b3 import get_efficientnet_b3
    from src.models.resnet18 import get_resnet18
    from src.models.resnet50 import get_resnet50

    if backbone == "resnet18":
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)
    if backbone == "resnet50":
        return get_resnet50(num_classes=num_classes, pretrained=pretrained)
    if backbone == "effb3":
        return get_efficientnet_b3(num_classes=num_classes, pretrained=pretrained)
    if backbone in _NEW_BUILDERS:
        return _NEW_BUILDERS[backbone](num_classes, pretrained)
    raise ValueError(f"Unknown backbone: {backbone}")
