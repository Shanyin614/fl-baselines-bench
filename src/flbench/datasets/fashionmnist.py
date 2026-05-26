"""FashionMNIST dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from flbench.core.client import FLClient
from flbench.core.types import DatasetInfo
from flbench.datasets.partitions import (
    DEFAULT_TRUE_GROUPS,
    build_dirichlet_client_metas,
    build_label_group_client_metas,
)
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
        f"seed{seed}.json"
    )


def _label_group_split_name(cfg, seed: int) -> str:
    return (
        f"label_group_major{float(cfg.dataset.major_ratio):.2f}_"
        f"clients{int(cfg.dataset.num_clients)}_seed{seed}.json"
    )


def build_fashionmnist_clients(cfg, device: torch.device) -> tuple[list[FLClient], DatasetInfo]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    data_root = Path(str(cfg.dataset.get("data_root", "data")))

    train_dataset = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    seed = int(cfg.runtime.get("seed", 42))
    partition = str(cfg.dataset.get("partition", "label_group")).lower()

    # CARES-lite used "manual" for the old hand-designed label-group partition.
    if partition == "manual":
        partition = "label_group"

    split_file_cfg = cfg.dataset.get("split_file", None)
    if split_file_cfg:
        split_file = Path(str(split_file_cfg))
    else:
        split_dir = Path(str(cfg.dataset.get("split_dir", "splits/fashionmnist")))
        if partition == "dirichlet":
            split_file = split_dir / _dirichlet_split_name(cfg, seed)
        elif partition == "label_group":
            split_file = split_dir / _label_group_split_name(cfg, seed)
        else:
            raise ValueError(
                f"unsupported FashionMNIST partition: {partition}. "
                "Use 'label_group', 'manual', or 'dirichlet'."
            )

    default_true_groups = [
        list(map(int, g))
        for g in cfg.dataset.get("true_groups", DEFAULT_TRUE_GROUPS)
    ]

    if split_file.exists():
        metas = load_client_metas(split_file)
        metadata = _read_split_metadata(split_file)
        true_groups = metadata.get("true_groups", default_true_groups)
        true_groups = [list(map(int, g)) for g in true_groups]
    else:
        if partition == "dirichlet":
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
        else:
            true_groups = default_true_groups
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
                "partition": partition,
                "num_clients": int(cfg.dataset.num_clients),
                "train_samples_per_client": int(cfg.dataset.train_samples_per_client),
                "test_samples_per_client": int(cfg.dataset.test_samples_per_client),
                "val_ratio": float(cfg.dataset.val_ratio),
                "num_classes": int(cfg.dataset.num_classes),
                "true_groups": true_groups,
                "major_ratio": float(cfg.dataset.get("major_ratio", 0.85)),
                "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
                "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
                "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
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
        extra={
            "num_clients": len(clients),
            "partition": partition,
            "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
            "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
            "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
        },
    )
