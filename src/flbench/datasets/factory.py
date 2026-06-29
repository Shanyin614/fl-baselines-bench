"""Dataset factory."""

from __future__ import annotations

import torch

from flbench.datasets.cifar10 import build_cifar10_clients
from flbench.datasets.fashionmnist import build_fashionmnist_clients
from flbench.datasets.tabular import build_tabular_clients


def build_clients(cfg, device: torch.device):
    name = str(cfg.dataset.name).lower()

    if name in {"fashionmnist", "fmnist"}:
        return build_fashionmnist_clients(cfg, device=device)

    if name == "cifar10":
        return build_cifar10_clients(cfg, device=device)

    if name in {
        "cicids2017",
        "cicids-2017",
        "cicids",
        "unsw_nb15",
        "unsw-nb15",
        "unsw",
    }:
        return build_tabular_clients(cfg, device=device)

    raise ValueError(f"unsupported dataset: {cfg.dataset.name}")
