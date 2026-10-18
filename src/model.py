import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


NUM_CLASSES = 8


def build_model(num_classes=NUM_CLASSES):
    # Load EfficientNet-B0 pretrained on ImageNet
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze the pretrained backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classifier
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


if __name__ == "__main__":
    model = build_model()

    print(model)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")