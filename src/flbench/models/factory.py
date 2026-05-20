"""Model factory."""
from __future__ import annotations

from typing import Callable

from torch import nn

from flbench.models.small_cnn import SmallCNN


def build_model_fn(cfg) -> Callable[[], nn.Module]:
    name = str(cfg.model.name).lower()
    num_classes = int(cfg.model.get("num_classes", cfg.dataset.get("num_classes", 10)))
    if name == "small_cnn":
        return lambda: SmallCNN(num_classes=num_classes)
    raise ValueError(f"unsupported model: {cfg.model.name}")
