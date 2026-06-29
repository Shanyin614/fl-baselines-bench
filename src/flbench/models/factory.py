"""Model factory."""

from __future__ import annotations

from typing import Callable

from torch import nn

from flbench.models.mlp import MLP
from flbench.models.small_cnn import SmallCNN


def build_model_fn(cfg, dataset_info=None) -> Callable[[], nn.Module]:
    name = str(cfg.model.name).lower()

    num_classes = int(cfg.model.get("num_classes", cfg.dataset.get("num_classes", 10)))
    input_channels = int(cfg.model.get("input_channels", 1))
    image_size = int(cfg.model.get("image_size", 28))

    input_dim = cfg.model.get("input_dim", None)
    if input_dim is None:
        input_dim = cfg.dataset.get("input_dim", None)
    if input_dim is None and dataset_info is not None:
        input_dim = dataset_info.extra.get("input_dim", None)
    input_dim = int(input_dim or 784)

    if name == "small_cnn":
        return lambda: SmallCNN(
            num_classes=num_classes,
            input_channels=input_channels,
            image_size=image_size,
        )

    if name == "mlp":
        return lambda: MLP(num_classes=num_classes, input_dim=input_dim)

    raise ValueError(f"unsupported model: {cfg.model.name}")
