# src/models/resnet50.py

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    ResNet-50 backbone cho bài journal.
    """
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1  # hoặc DEFAULT
    else:
        weights = None

    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
