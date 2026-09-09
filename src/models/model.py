"""Définition du modèle CNN."""
import torch.nn as nn
from torchvision import models


def build_model(backbone: str, n_classes: int, pretrained: bool = True):
    if backbone == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, n_classes)

        return model

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)

        return model

    raise ValueError(f"Backbone inconnu : {backbone}")


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        is_head = name.startswith("classifier") or name.startswith("fc")
        if not is_head:
            param.requires_grad = trainable
