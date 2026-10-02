"""Define the MobileNetV3 Small binary image classifier."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


MODEL_NAME = "MobileNetV3 Small"
PRETRAINED_WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT


def create_model(
    num_classes: int = 2,
    freeze_features: bool = True,
    unfreeze_final_feature_block: bool = False,
    weights: MobileNet_V3_Small_Weights | None = PRETRAINED_WEIGHTS,
) -> nn.Module:
    """Create MobileNetV3 Small with configurable controlled fine-tuning."""
    model = mobilenet_v3_small(weights=weights)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        if unfreeze_final_feature_block:
            for parameter in model.features[-1].parameters():
                parameter.requires_grad = True

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model


def main() -> None:
    device = torch.device("cpu")
    model = create_model(
        num_classes=2,
        freeze_features=True,
        unfreeze_final_feature_block=True,
    ).to(device)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameters = total_parameters - trainable_parameters
    frozen_feature_blocks = all(
        not parameter.requires_grad
        for block in model.features[:-1]
        for parameter in block.parameters()
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    expected_trainable_parts = all(
        name.startswith("features.12.") or name.startswith("classifier.")
        for name in trainable_names
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Frozen parameters: {frozen_parameters:,}")
    print(f"Frozen feature blocks unchanged: {frozen_feature_blocks}")
    print("Unfrozen feature block: features[12]")
    print(f"Only final feature block and classifier trainable: {expected_trainable_parts}")
    print("Trainable parameter names:")
    for name in trainable_names:
        print(f"  {name}")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
