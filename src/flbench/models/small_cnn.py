"""SmallCNN aligned with CARES-lite experiments."""

from __future__ import annotations

import torch.nn as nn

LAST_LAYER_PREFIX = "classifier.2"


class SmallCNN(nn.Module):
    """Lightweight CNN for 28x28 grayscale or 32x32 RGB images."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        image_size: int = 28,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(int(input_channels), 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        pooled_size = int(image_size) // 4

        self.classifier = nn.Sequential(
            nn.Linear(32 * pooled_size * pooled_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
