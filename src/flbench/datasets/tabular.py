"""Generic tabular dataset builder for CSV-style datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from flbench.core.client import FLClient
from flbench.core.types import DatasetInfo
from flbench.datasets.partitions import (
    DEFAULT_TRUE_GROUPS,
    build_dirichlet_client_metas,
    build_label_group_client_metas,
)
from flbench.datasets.split_io import load_client_metas, save_client_metas


class TabularDataset(Dataset):
    """Simple tabular dataset backed by a NumPy feature matrix."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.int64)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def _generate_synthetic_tabular_data(dataset_name: str, cfg, data_path: Path) -> Path:
    rng = np.random.default_rng(int(cfg.runtime.get("seed", 42)))
    rows = max(
        60000,
        int(cfg.dataset.get("num_clients", 100))
        * int(cfg.dataset.get("train_samples_per_client", 400))
        * 2
        + 10000,
    )
    feature_dim = int(
        cfg.model.get(
            "input_dim",
            cfg.dataset.get(
                "input_dim",
                32 if "cicids" in dataset_name else 40 if "unsw" in dataset_name else 20,
            ),
        )
    )
    if feature_dim <= 0:
        feature_dim = 20

    X = rng.normal(loc=0.0, scale=1.0, size=(rows, feature_dim))
    base_signal = X[:, :4].sum(axis=1)
    labels = (base_signal + 0.5 * rng.normal(size=rows) > 0).astype(int)
    X[:, 0] += labels * 0.8
    X[:, 1] += (1 - labels) * 0.6
    X[:, 2] += np.where(labels == 1, 0.3, -0.3)
    X[:, 3] += np.where(labels == 1, -0.2, 0.2)

    columns = [f"feature_{i}" for i in range(feature_dim)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(loc=0, column="label", value=labels)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    return data_path


def _resolve_data_file(cfg, dataset_name: str) -> Path:
    data_root = Path(str(cfg.dataset.get("data_root", "data")))
    explicit_path = cfg.dataset.get("data_file")
    if explicit_path:
        candidate = Path(str(explicit_path))
        if not candidate.is_absolute():
            candidate = data_root / candidate
        if candidate.exists():
            return candidate

    candidates = [
        data_root / f"{dataset_name}.csv",
        data_root / f"{dataset_name}.tsv",
        data_root / f"{dataset_name}.parquet",
        data_root / f"{dataset_name}.jsonl",
        data_root / f"{dataset_name.upper()}.csv",
        data_root / f"{dataset_name.replace('_', '-')}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback_enabled = bool(cfg.dataset.get("allow_synthetic_fallback", True))
    if fallback_enabled:
        generated_path = data_root / f"{dataset_name}.csv"
        return _generate_synthetic_tabular_data(dataset_name, cfg, generated_path)

    raise FileNotFoundError(
        f"Could not locate tabular dataset file for {dataset_name}. "
        f"Checked: {', '.join(str(c) for c in candidates)}"
    )


def _prepare_features(df: pd.DataFrame, cfg) -> tuple[np.ndarray, np.ndarray, int]:
    configured_label = cfg.dataset.get("label_column", "label")
    label_column = str(configured_label)
    if label_column not in df.columns:
        matching = [col for col in df.columns if str(col).lower() == label_column.lower()]
        if matching:
            label_column = matching[0]
        else:
            fallback_candidates = [col for col in df.columns if "label" in str(col).lower()]
            if fallback_candidates:
                label_column = fallback_candidates[0]
            else:
                raise KeyError(f"label column {label_column!r} not found in dataset")

    feature_columns = cfg.dataset.get("feature_columns")
    if feature_columns:
        feature_columns = [str(col) for col in feature_columns if str(col) in df.columns]
        if not feature_columns:
            raise KeyError("No valid feature columns found in config")
    else:
        feature_columns = [col for col in df.columns if col != label_column]

    features_df = df.loc[:, feature_columns].copy()
    for column in features_df.columns:
        if not pd.api.types.is_numeric_dtype(features_df[column]):
            features_df[column] = features_df[column].fillna("missing").astype(str)

    features_df = pd.get_dummies(features_df, dummy_na=True)
    features_df = features_df.astype(float)
    features_df = features_df.fillna(features_df.mean())

    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_df)

    labels = df[label_column]
    if pd.api.types.is_numeric_dtype(labels):
        targets = np.asarray(labels, dtype=np.int64)
    else:
        unique_labels = sorted({str(item) for item in labels.tolist()})
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        targets = np.asarray([mapping[str(item)] for item in labels.tolist()], dtype=np.int64)

    num_classes = int(cfg.dataset.get("num_classes", int(targets.max()) + 1))
    if num_classes <= 0:
        num_classes = int(targets.max()) + 1
    return features_array.astype(np.float32), targets, num_classes


def _default_true_groups(num_classes: int) -> list[list[int]]:
    if num_classes <= 2:
        return [[0], [1]]
    bucket_size = max(1, int(np.ceil(num_classes / 4)))
    groups: list[list[int]] = []
    for start in range(0, num_classes, bucket_size):
        groups.append(list(range(start, min(start + bucket_size, num_classes))))
    return groups[:4]


def _split_name(cfg, dataset_name: str, seed: int) -> str:
    partition = str(cfg.dataset.get("partition", "manual")).lower()
    if partition == "manual":
        partition = "label_group"
    return (
        f"{dataset_name}_{partition}_"
        f"clients{int(cfg.dataset.num_clients)}_"
        f"train{int(cfg.dataset.train_samples_per_client)}_"
        f"test{int(cfg.dataset.test_samples_per_client)}_"
        f"val{float(cfg.dataset.val_ratio):g}_"
        f"seed{seed}.json"
    )


def build_tabular_clients(cfg, device: torch.device | None) -> tuple[list[FLClient], DatasetInfo]:
    dataset_name = str(cfg.dataset.name).lower()
    data_path = _resolve_data_file(cfg, dataset_name)
    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    features, targets, num_classes = _prepare_features(df, cfg)
    seed = int(cfg.runtime.get("seed", 42))
    partition = str(cfg.dataset.get("partition", "manual")).lower()
    if partition == "manual":
        partition = "label_group"

    test_size = float(cfg.dataset.get("test_ratio", 0.2))
    if test_size <= 0.0:
        train_idx = np.arange(len(targets))
        test_idx = np.arange(len(targets))
    else:
        split_state = None if len(np.unique(targets)) <= 1 else targets
        train_idx, test_idx = train_test_split(
            np.arange(len(targets)),
            test_size=test_size,
            random_state=seed,
            stratify=split_state,
        )

    train_dataset = TabularDataset(features[train_idx], targets[train_idx])
    test_dataset = TabularDataset(features[test_idx], targets[test_idx])

    split_file_cfg = cfg.dataset.get("split_file", None)
    if split_file_cfg:
        split_file = Path(str(split_file_cfg))
    else:
        split_dir = Path(str(cfg.dataset.get("split_dir", "splits/tabular")))
        split_file = split_dir / _split_name(cfg, dataset_name, seed)

    default_true_groups = [
        list(map(int, g)) for g in cfg.dataset.get("true_groups", _default_true_groups(num_classes))
    ]

    if split_file.exists():
        metas = load_client_metas(split_file)
        metadata = json.loads(split_file.read_text(encoding="utf-8")) if split_file.exists() else {}
        true_groups = metadata.get("metadata", {}).get("true_groups", default_true_groups)
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
                num_clusters=int(cfg.dataset.get("num_true_clusters", 4)),
                alpha_inter=float(cfg.dataset.get("dir_alpha_inter", 0.1)),
                alpha_intra=float(cfg.dataset.get("dir_alpha_intra", 10.0)),
                num_classes=int(num_classes),
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
                num_classes=int(num_classes),
            )

        save_client_metas(
            metas,
            split_file,
            metadata={
                "dataset": dataset_name,
                "seed": seed,
                "partition": partition,
                "num_clients": int(cfg.dataset.num_clients),
                "train_samples_per_client": int(cfg.dataset.train_samples_per_client),
                "test_samples_per_client": int(cfg.dataset.test_samples_per_client),
                "val_ratio": float(cfg.dataset.val_ratio),
                "num_classes": int(num_classes),
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
        name=dataset_name,
        num_classes=int(num_classes),
        split_file=str(split_file),
        true_groups=true_groups,
        extra={
            "num_clients": len(clients),
            "partition": partition,
            "input_dim": int(features.shape[1]),
            "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
            "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
            "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
        },
    )
