"""Dataset factory."""
from __future__ import annotations

import torch

from flbench.datasets.fashionmnist import build_fashionmnist_clients


def build_clients(cfg, device: torch.device):
    name = str(cfg.dataset.name).lower()
    if name == "fashionmnist":
        return build_fashionmnist_clients(cfg, device=device)
    raise ValueError(f"unsupported dataset: {cfg.dataset.name}")
