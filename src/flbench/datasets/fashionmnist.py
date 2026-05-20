"""FashionMNIST dataset builder."""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms

from flbench.core.client import FLClient
from flbench.core.types import DatasetInfo
from flbench.datasets.partitions import DEFAULT_TRUE_GROUPS, build_label_group_client_metas
from flbench.datasets.split_io import load_client_metas, save_client_metas


def build_fashionmnist_clients(cfg, device: torch.device) -> tuple[list[FLClient], DatasetInfo]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    data_root = Path(str(cfg.dataset.get("data_root", "data")))
    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    split_file_cfg = cfg.dataset.get("split_file", None)
    seed = int(cfg.runtime.get("seed", 42))
    true_groups = [list(map(int, g)) for g in cfg.dataset.get("true_groups", DEFAULT_TRUE_GROUPS)]
    if split_file_cfg:
        split_file = Path(str(split_file_cfg))
    else:
        split_dir = Path(str(cfg.dataset.get("split_dir", "splits/fashionmnist")))
        split_file = split_dir / (
            f"label_group_major{float(cfg.dataset.major_ratio):.2f}_"
            f"clients{int(cfg.dataset.num_clients)}_seed{seed}.json"
        )

    if split_file.exists():
        metas = load_client_metas(split_file)
    else:
        metas = build_label_group_client_metas(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_clients=int(cfg.dataset.num_clients),
            train_samples_per_client=int(cfg.dataset.train_samples_per_client),
            test_samples_per_client=int(cfg.dataset.test_samples_per_client),
            major_ratio=float(cfg.dataset.major_ratio),
            val_ratio=float(cfg.dataset.val_ratio),
            seed=seed,
            true_groups=true_groups,
            num_classes=int(cfg.dataset.num_classes),
        )
        save_client_metas(
            metas,
            split_file,
            metadata={
                "dataset": "fashionmnist",
                "seed": seed,
                "partition": str(cfg.dataset.partition),
                "major_ratio": float(cfg.dataset.major_ratio),
                "true_groups": true_groups,
            },
        )

    clients = [
        FLClient(
            meta=meta,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=int(cfg.train.batch_size),
            num_workers=int(cfg.dataset.get("num_workers", 0)),
            device=device,
        )
        for meta in metas
    ]
    return clients, DatasetInfo(
        name="fashionmnist",
        num_classes=int(cfg.dataset.num_classes),
        split_file=str(split_file),
        true_groups=true_groups,
        extra={"num_clients": len(clients)},
    )
