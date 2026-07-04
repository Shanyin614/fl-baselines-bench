"""Generic tabular dataset builder for explicit train/test NIDS CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from flbench.core.client import FLClient
from flbench.core.types import DatasetInfo
from flbench.datasets.partitions import (
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


def _resolve_configured_path(cfg, key: str, *, required: bool = True) -> Path | None:
    value = cfg.dataset.get(key, None)
    if value in (None, ""):
        if required:
            raise ValueError(f"dataset.{key} must be provided")
        return None

    path = Path(str(value))
    if not path.is_absolute():
        root = Path(str(cfg.dataset.get("data_root", "data")))
        path = root / path
    return path


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find dataset file: {path}\n"
            "If CARES-Lite and fl-baselines-bench are sibling directories, create symlinks first:\n"
            "  mkdir -p data\n"
            "  ln -s ../CARES-lite/data/unsw data/unsw\n"
            "  ln -s ../CARES-lite/data/cicids2017 data/cicids2017"
        )
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_label_column(df: pd.DataFrame, configured: str | None) -> str:
    candidates: list[str] = []
    if configured:
        candidates.append(str(configured))
        candidates.append(str(configured).lower())
        candidates.append(str(configured).upper())
    candidates.extend(["label", "Label", "target", "Target", "class", "Class"])
    for col in candidates:
        if col in df.columns:
            return col

    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if str(name).lower() in lower:
            return lower[str(name).lower()]

    fallback = [c for c in df.columns if "label" in str(c).lower()]
    if fallback:
        return fallback[0]
    raise KeyError(f"Could not find label column. Configured={configured!r}, columns={list(df.columns)[:20]}")


def _normalise_binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    text = series.astype(str).str.strip()
    lower = text.str.lower()
    benign = {"0", "normal", "benign", "background", "none"}
    attack = {"1", "attack", "attacks", "malicious", "anomaly", "abnormal"}
    values = set(lower.dropna().unique().tolist())
    if values and values.issubset(benign | attack):
        return lower.map(lambda x: 0 if x in benign else 1).astype(int)
    return text


def _encode_targets(train_raw: pd.Series, test_raw: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    train_norm = _normalise_binary_labels(train_raw)
    test_norm = _normalise_binary_labels(test_raw)

    if pd.api.types.is_numeric_dtype(train_norm) and pd.api.types.is_numeric_dtype(test_norm):
        train_y = train_norm.astype(int).to_numpy()
        test_y = test_norm.astype(int).to_numpy()
        labels = sorted(set(train_y.tolist()) | set(test_y.tolist()))
        mapping = {str(label): int(label) for label in labels}
        return train_y, test_y, mapping

    train_text = train_norm.astype(str)
    test_text = test_norm.astype(str)
    labels = sorted(set(train_text.tolist()) | set(test_text.tolist()))
    mapping = {label: idx for idx, label in enumerate(labels)}
    train_y = train_text.map(mapping).astype(int).to_numpy()
    test_y = test_text.map(mapping).astype(int).to_numpy()
    return train_y, test_y, mapping


def _feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, train_label: str, test_label: str, cfg) -> list[str]:
    configured = cfg.dataset.get("feature_columns", None)

    if configured:
        cols = [str(c) for c in configured]
        missing = [c for c in cols if c not in train_df.columns or c not in test_df.columns]
        if missing:
            raise KeyError(f"Configured feature columns missing from train/test CSV: {missing}")
    else:
        cols = []
        for col in train_df.columns:
            if col == train_label:
                continue
            if col in test_df.columns and col != test_label:
                cols.append(col)

    drop_columns_cfg = cfg.dataset.get("drop_columns", [])
    drop_exact = {str(col) for col in drop_columns_cfg}
    drop_lower = {str(col).lower() for col in drop_columns_cfg}
    auto_drop_contains = [str(x).lower() for x in cfg.dataset.get("auto_drop_if_contains", [])]

    kept_cols = []
    dropped_cols = []

    for col in cols:
        col_str = str(col)
        col_lower = col_str.lower()

        should_drop = (
            col_str in drop_exact
            or col_lower in drop_lower
            or any(token in col_lower for token in auto_drop_contains)
        )

        if should_drop:
            dropped_cols.append(col)
        else:
            kept_cols.append(col)

    if dropped_cols:
        print(f"[dataset] dropped feature columns: {dropped_cols}")

    if not kept_cols:
        raise KeyError("No usable feature columns found after excluding labels and drop_columns.")

    return kept_cols


def _prepare_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, int], int]:
    label_cfg = cfg.dataset.get("label_column", "label")
    train_label = _resolve_label_column(train_df, label_cfg)
    test_label = _resolve_label_column(test_df, label_cfg)
    cols = _feature_columns(train_df, test_df, train_label, test_label, cfg)

    train_y, test_y, label_mapping = _encode_targets(train_df[train_label], test_df[test_label])

    train_x_raw = train_df.loc[:, cols].copy()
    test_x_raw = test_df.loc[:, cols].copy()

    train_x = pd.get_dummies(train_x_raw, dummy_na=True)
    test_x = pd.get_dummies(test_x_raw, dummy_na=True)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)

    train_x = train_x.apply(pd.to_numeric, errors="coerce")
    test_x = test_x.apply(pd.to_numeric, errors="coerce")
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)

    means = train_x.mean(axis=0).fillna(0.0)
    train_x = train_x.fillna(means)
    test_x = test_x.fillna(means)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_x.to_numpy(dtype=np.float32)).astype(np.float32)
    test_features = scaler.transform(test_x.to_numpy(dtype=np.float32)).astype(np.float32)

    inferred_classes = int(max(train_y.max(initial=0), test_y.max(initial=0)) + 1)
    cfg_classes = int(cfg.dataset.get("num_classes", inferred_classes))
    num_classes = max(cfg_classes, inferred_classes)
    input_dim = int(train_features.shape[1])
    return train_features, train_y, test_features, test_y, num_classes, label_mapping, input_dim


def _default_true_groups(num_classes: int, num_clusters: int) -> list[list[int]]:
    if num_clusters <= 1:
        return [list(range(num_classes))]
    groups = [[] for _ in range(int(num_clusters))]
    for y in range(int(num_classes)):
        groups[y % int(num_clusters)].append(int(y))
    return groups


def _split_name(cfg, dataset_name: str, seed: int, explicit_test: bool) -> str:
    partition = str(cfg.dataset.get("partition", "manual")).lower()
    if partition == "manual":
        partition = "label_group"
    source = "explicit" if explicit_test else "split"
    return (
        f"{dataset_name}_{partition}_{source}_"
        f"clients{int(cfg.dataset.num_clients)}_"
        f"train{int(cfg.dataset.train_samples_per_client)}_"
        f"test{int(cfg.dataset.test_samples_per_client)}_"
        f"val{float(cfg.dataset.val_ratio):g}_"
        f"seed{seed}.json"
    )


def build_tabular_clients(cfg, device: torch.device | None) -> tuple[list[FLClient], DatasetInfo]:
    dataset_name = str(cfg.dataset.name).lower()
    seed = int(cfg.runtime.get("seed", 42))

    train_path = _resolve_configured_path(cfg, "data_file", required=True)
    test_path = _resolve_configured_path(cfg, "test_file", required=False)

    if test_path is not None:
        train_df = _read_frame(train_path)
        test_df = _read_frame(test_path)
        split_source = "explicit_train_test"
    else:
        full_df = _read_frame(train_path)
        label_col = _resolve_label_column(full_df, cfg.dataset.get("label_column", "label"))
        test_size = float(cfg.dataset.get("test_ratio", 0.2))
        try:
            stratify = full_df[label_col] if full_df[label_col].nunique() > 1 else None
            train_df, test_df = train_test_split(
                full_df,
                test_size=test_size,
                random_state=seed,
                stratify=stratify,
            )
        except ValueError:
            train_df, test_df = train_test_split(full_df, test_size=test_size, random_state=seed, stratify=None)
        split_source = f"single_file_train_test_split_{test_size:g}"

    train_features, train_targets, test_features, test_targets, num_classes, label_mapping, input_dim = _prepare_train_test_features(
        train_df, test_df, cfg
    )

    train_dataset = TabularDataset(train_features, train_targets)
    test_dataset = TabularDataset(test_features, test_targets)

    split_file_cfg = cfg.dataset.get("split_file", None)
    if split_file_cfg:
        split_file = Path(str(split_file_cfg))
    else:
        split_dir = Path(str(cfg.dataset.get("split_dir", "splits/tabular")))
        split_file = split_dir / _split_name(cfg, dataset_name, seed, explicit_test=(test_path is not None))

    default_true_groups = [
        list(map(int, g))
        for g in cfg.dataset.get(
            "true_groups",
            _default_true_groups(num_classes, int(cfg.dataset.get("num_true_clusters", 2))),
        )
    ]

    if split_file.exists():
        metas = load_client_metas(split_file)
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        true_groups = payload.get("metadata", {}).get("true_groups", default_true_groups)
        true_groups = [list(map(int, g)) for g in true_groups]
        print(f"[dataset] loaded split metadata from {split_file}")
    else:
        partition = str(cfg.dataset.get("partition", "manual")).lower()
        if partition == "dirichlet":
            metas, true_groups = build_dirichlet_client_metas(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_clients=int(cfg.dataset.num_clients),
                train_samples_per_client=int(cfg.dataset.train_samples_per_client),
                test_samples_per_client=int(cfg.dataset.test_samples_per_client),
                val_ratio=float(cfg.dataset.val_ratio),
                seed=seed,
                num_clusters=int(cfg.dataset.get("num_true_clusters", 2)),
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
                major_ratio=float(cfg.dataset.get("major_ratio", 0.85)),
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
                "split_source": split_source,
                "partition": str(cfg.dataset.get("partition", "manual")).lower(),
                "num_clients": int(cfg.dataset.num_clients),
                "train_samples_per_client": int(cfg.dataset.train_samples_per_client),
                "test_samples_per_client": int(cfg.dataset.test_samples_per_client),
                "val_ratio": float(cfg.dataset.val_ratio),
                "num_classes": int(num_classes),
                "true_groups": true_groups,
                "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
                "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
                "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
            },
        )
        print(f"[dataset] saved split metadata to {split_file}")

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

    print("\n[NIDS] Dataset preprocessing summary")
    print(f"  dataset: {dataset_name}")
    print(f"  split_source: {split_source}")
    print(f"  train_file: {train_path}")
    if test_path is not None:
        print(f"  test_file: {test_path}")
    print(f"  train shape: {train_df.shape}")
    print(f"  test shape: {test_df.shape}")
    print(f"  input_dim: {input_dim}")
    print(f"  num_classes: {num_classes}")
    print(f"  label_map: {label_mapping}")

    return clients, DatasetInfo(
        name=dataset_name,
        num_classes=int(num_classes),
        split_file=str(split_file),
        true_groups=true_groups,
        extra={
            "num_clients": len(clients),
            "partition": str(cfg.dataset.get("partition", "manual")).lower(),
            "input_dim": int(input_dim),
            "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
            "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
            "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
            "split_source": split_source,
            "label_mapping": label_mapping,
        },
    )

