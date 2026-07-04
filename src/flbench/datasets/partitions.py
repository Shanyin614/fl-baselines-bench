"""CARES-compatible FashionMNIST client partitions."""

from __future__ import annotations

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
    major_set = set(map(int, major_labels))
    bg = [int(y) for y in range(int(num_classes)) if int(y) not in major_set]

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
    """Original label-group non-IID partition."""
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


def _sample_by_label_probs(
    pools: dict[int, np.ndarray],
    probs: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> list[int]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()

    labels = rng.choice(np.arange(len(probs)), size=int(n), p=probs)

    sampled: list[int] = []
    for y in labels:
        sampled.append(int(rng.choice(pools[int(y)])))

    rng.shuffle(sampled)
    return sampled


def build_dirichlet_client_metas(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int,
    train_samples_per_client: int,
    test_samples_per_client: int,
    val_ratio: float,
    seed: int,
    num_clusters: int = 10,
    alpha_inter: float = 0.1,
    alpha_intra: float = 10.0,
    num_classes: int = 10,
) -> tuple[list[ClientMeta], list[list[int]]]:
    """
    CARES-lite DPMM-CFL-style two-level Dirichlet partition.

    Level 1:
      Each ground-truth cluster k has label distribution pi_k ~ Dir(alpha_inter).

    Level 2:
      Each client i in cluster k has theta_i ~ Dir(alpha_intra * pi_k).

    group_id is the ground-truth cluster id, used only for ARI/NMI/Purity.
    """
    rng = np.random.default_rng(int(seed))

    train_pools = label_to_indices(train_dataset, num_classes=num_classes)
    test_pools = label_to_indices(test_dataset, num_classes=num_classes)

    cluster_priors = rng.dirichlet(
        alpha=np.full(int(num_classes), float(alpha_inter), dtype=np.float64),
        size=int(num_clusters),
    )

    metas: list[ClientMeta] = []

    for cid in range(int(num_clients)):
        gid = int(cid % int(num_clusters))
        pi_k = cluster_priors[gid]

        theta_i = rng.dirichlet(float(alpha_intra) * pi_k + 1e-6)

        all_train = _sample_by_label_probs(
            train_pools,
            theta_i,
            int(train_samples_per_client),
            rng,
        )

        n_val = max(1, int(round(len(all_train) * float(val_ratio))))
        val_indices = all_train[:n_val]
        train_indices = all_train[n_val:]

        test_indices = _sample_by_label_probs(
            test_pools,
            theta_i,
            int(test_samples_per_client),
            rng,
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

    true_groups = [
        [int(x) for x in np.argsort(-cluster_priors[k])[:3]]
        for k in range(int(num_clusters))
    ]

    return metas, true_groups

