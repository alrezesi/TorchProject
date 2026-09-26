"""
CNN architecture for Fashion-MNIST classification.

Three convolutional blocks (64 -> 128 -> 256 channels), each block:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout
followed by a dense classification head.
"""

import torch
import torch.nn as nn


class FashionCNN(nn.Module):
    """
    Input shape:  (batch_size, 1, 28, 28)
    Output shape: (batch_size, num_classes) -> raw logits

    Spatial size shrinks after each block's MaxPool2d:
        28x28 -> 14x14 -> 7x7 -> 3x3 (integer division, floor)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 64 channels, 28x28 -> 14x14
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.20),

            # Block 2: 64 -> 128 channels, 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 3: 128 -> 256 channels, 7x7 -> 3x3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 256 channels * 3 * 3 spatial size = 2304 features
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.40),

            # Final layer: raw logits (nn.CrossEntropyLoss applies log-softmax internally)
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FashionCNN()
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    dummy_batch = torch.randn(32, 1, 28, 28)
    output = model(dummy_batch)
    print(f"Input shape:  {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")  # expected: (32, 10)
    assert output.shape == (32, 10), "Output shape mismatch!"
    print("Shape check passed.")