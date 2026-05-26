"""CIFAR-10 dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from flbench.core.client import FLClient
from flbench.core.types import DatasetInfo
from flbench.datasets.partitions import build_dirichlet_client_metas
from flbench.datasets.split_io import load_client_metas, save_client_metas


def _read_split_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return dict(payload.get("metadata", {}))
    except Exception:
        return {}


def _dirichlet_split_name(cfg, seed: int) -> str:
    return (
        f"dirichlet_k{int(cfg.dataset.get('num_true_clusters', 10))}_"
        f"ainter{float(cfg.dataset.get('dir_alpha_inter', 0.1)):g}_"
        f"aintra{float(cfg.dataset.get('dir_alpha_intra', 10.0)):g}_"
        f"clients{int(cfg.dataset.num_clients)}_"
        f"train{int(cfg.dataset.train_samples_per_client)}_"
        f"test{int(cfg.dataset.test_samples_per_client)}_"
        f"val{float(cfg.dataset.val_ratio):g}_"
        f"seed{seed}.json"
    )


def build_cifar10_clients(cfg, device: torch.device) -> tuple[list[FLClient], DatasetInfo]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    data_root = Path(str(cfg.dataset.get("data_root", "data")))

    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    seed = int(cfg.runtime.get("seed", 42))
    partition = str(cfg.dataset.get("partition", "dirichlet")).lower()

    if partition != "dirichlet":
        raise ValueError(
            f"CIFAR-10 paper experiment should use partition='dirichlet', got: {partition}"
        )

    split_file_cfg = cfg.dataset.get("split_file", None)
    if split_file_cfg:
        split_file = Path(str(split_file_cfg))
    else:
        split_dir = Path(str(cfg.dataset.get("split_dir", "splits/cifar10")))
        split_file = split_dir / _dirichlet_split_name(cfg, seed)

    if split_file.exists():
        metas = load_client_metas(split_file)
        metadata = _read_split_metadata(split_file)
        true_groups = metadata.get("true_groups", [])
        true_groups = [list(map(int, g)) for g in true_groups]
    else:
        metas, true_groups = build_dirichlet_client_metas(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_clients=int(cfg.dataset.num_clients),
            train_samples_per_client=int(cfg.dataset.train_samples_per_client),
            test_samples_per_client=int(cfg.dataset.test_samples_per_client),
            val_ratio=float(cfg.dataset.val_ratio),
            seed=seed,
            num_clusters=int(cfg.dataset.get("num_true_clusters", 10)),
            alpha_inter=float(cfg.dataset.get("dir_alpha_inter", 0.1)),
            alpha_intra=float(cfg.dataset.get("dir_alpha_intra", 10.0)),
            num_classes=int(cfg.dataset.num_classes),
        )

        save_client_metas(
            metas,
            split_file,
            metadata={
                "dataset": "cifar10",
                "seed": seed,
                "partition": partition,
                "num_clients": int(cfg.dataset.num_clients),
                "train_samples_per_client": int(cfg.dataset.train_samples_per_client),
                "test_samples_per_client": int(cfg.dataset.test_samples_per_client),
                "val_ratio": float(cfg.dataset.val_ratio),
                "num_classes": int(cfg.dataset.num_classes),
                "num_true_clusters": int(cfg.dataset.get("num_true_clusters", 10)),
                "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
                "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
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
        name="cifar10",
        num_classes=int(cfg.dataset.num_classes),
        split_file=str(split_file),
        true_groups=true_groups,
        extra={
            "num_clients": len(clients),
            "partition": partition,
            "num_true_clusters": int(cfg.dataset.get("num_true_clusters", 10)),
            "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
            "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
            "input_channels": 3,
            "image_size": 32,
        },
    )
