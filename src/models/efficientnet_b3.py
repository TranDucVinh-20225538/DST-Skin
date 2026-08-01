import torch.nn as nn
import torchvision.models as models

def get_efficientnet_b3(num_classes=2, pretrained=True):
    if pretrained:
        weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
    else:
        model = models.efficientnet_b3()

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model