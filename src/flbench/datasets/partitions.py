"""CARES-compatible FashionMNIST label-group partitions."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from flbench.core.types import ClientMeta

DEFAULT_TRUE_GROUPS: list[list[int]] = [
    [0, 2, 6],
    [1, 3],
    [4, 8],
    [5, 7, 9],
]


def dataset_targets(dataset: Dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("dataset must expose a `.targets` attribute")
    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu().numpy().astype(int)
    return np.asarray(targets, dtype=int)


def label_to_indices(dataset: Dataset, num_classes: int) -> dict[int, np.ndarray]:
    targets = dataset_targets(dataset)
    return {int(y): np.where(targets == int(y))[0] for y in range(int(num_classes))}


def _sample_mixture(
    pools: dict[int, np.ndarray],
    major_labels: list[int],
    n: int,
    major_ratio: float,
    rng: np.random.Generator,
    num_classes: int,
) -> list[int]:
    bg = [int(y) for y in range(int(num_classes)) if int(y) not in set(map(int, major_labels))]
    n_major = int(round(int(n) * float(major_ratio)))
    sampled: list[int] = []
    for _ in range(n_major):
        y = int(rng.choice(major_labels))
        sampled.append(int(rng.choice(pools[y])))
    for _ in range(int(n) - n_major):
        y = int(rng.choice(bg))
        sampled.append(int(rng.choice(pools[y])))
    rng.shuffle(sampled)
    return sampled


def build_label_group_client_metas(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    train_samples_per_client: int,
    test_samples_per_client: int,
    major_ratio: float,
    val_ratio: float,
    seed: int,
    true_groups: list[list[int]] | None = None,
    num_classes: int = 10,
) -> list[ClientMeta]:
    """Build the same style of label-group non-IID clients used by CARES-lite."""
    true_groups = true_groups or DEFAULT_TRUE_GROUPS
    rng = np.random.default_rng(int(seed))
    train_pools = label_to_indices(train_dataset, num_classes=num_classes)
    test_pools = label_to_indices(test_dataset, num_classes=num_classes)

    metas: list[ClientMeta] = []
    for cid in range(int(num_clients)):
        gid = int(cid % len(true_groups))
        all_train = _sample_mixture(
            train_pools,
            true_groups[gid],
            int(train_samples_per_client),
            float(major_ratio),
            rng,
            int(num_classes),
        )
        n_val = max(1, int(round(len(all_train) * float(val_ratio))))
        val_indices = all_train[:n_val]
        train_indices = all_train[n_val:]
        test_indices = _sample_mixture(
            test_pools,
            true_groups[gid],
            int(test_samples_per_client),
            float(major_ratio),
            rng,
            int(num_classes),
        )
        metas.append(
            ClientMeta(
                client_id=cid,
                group_id=gid,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )
    return metas
