"""Simple MLP model for tabular datasets."""

from __future__ import annotations

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_classes: int = 2, input_dim: int = 784) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 64),
            nn.ReLU(),
            nn.Linear(64, int(num_classes)),
        )

    def forward(self, x):
        return self.net(x)

