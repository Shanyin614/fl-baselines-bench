from pathlib import Path
import re

p = Path("src/flbench/datasets/tabular.py")
s = p.read_text(encoding="utf-8")

new_prepare = r'''
def _resolve_label_column(df: pd.DataFrame, cfg) -> str:
    configured_label = cfg.dataset.get("label_column", "label")
    label_column = str(configured_label)

    if label_column in df.columns:
        return label_column

    matching = [col for col in df.columns if str(col).lower() == label_column.lower()]
    if matching:
        return matching[0]

    fallback_candidates = [col for col in df.columns if "label" in str(col).lower()]
    if fallback_candidates:
        return fallback_candidates[0]

    raise KeyError(f"label column {label_column!r} not found in dataset")


def _select_feature_columns(df: pd.DataFrame, cfg, label_column: str) -> list:
    configured_features = cfg.dataset.get("feature_columns")

    if configured_features:
        feature_columns = [col for col in configured_features if str(col) in df.columns and str(col) != label_column]
        if not feature_columns:
            raise KeyError("No valid feature columns found in config")
    else:
        feature_columns = [col for col in df.columns if col != label_column]

    drop_columns_cfg = cfg.dataset.get("drop_columns", [])
    drop_columns_exact = {str(col) for col in drop_columns_cfg}
    drop_columns_lower = {str(col).lower() for col in drop_columns_cfg}
    auto_drop_contains = [str(x).lower() for x in cfg.dataset.get("auto_drop_if_contains", [])]

    kept_columns = []
    dropped_columns = []

    for col in feature_columns:
        col_str = str(col)
        col_lower = col_str.lower()

        should_drop = (
            col_str in drop_columns_exact
            or col_lower in drop_columns_lower
            or any(token in col_lower for token in auto_drop_contains)
        )

        if should_drop:
            dropped_columns.append(col)
        else:
            kept_columns.append(col)

    if dropped_columns:
        print(f"[dataset] dropped feature columns: {dropped_columns}")

    if not kept_columns:
        raise KeyError("No feature columns remain after applying drop_columns/auto_drop_if_contains")

    return kept_columns


def _prepare_features(
    df: pd.DataFrame,
    cfg,
    reference_columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
    label_mapping: dict[str, int] | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, list[str], StandardScaler, dict[str, int] | None]:
    label_column = _resolve_label_column(df, cfg)
    feature_columns = _select_feature_columns(df, cfg, label_column)

    features_df = df.loc[:, feature_columns].copy()

    for column in features_df.columns:
        if not pd.api.types.is_numeric_dtype(features_df[column]):
            features_df[column] = features_df[column].fillna("missing").astype(str)

    features_df = pd.get_dummies(features_df, dummy_na=True)
    features_df = features_df.astype(float)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(features_df.mean(numeric_only=True)).fillna(0.0)

    if fit or reference_columns is None:
        reference_columns = list(features_df.columns)
    else:
        features_df = features_df.reindex(columns=reference_columns, fill_value=0.0)

    if fit or scaler is None:
        scaler = StandardScaler()
        features_array = scaler.fit_transform(features_df)
    else:
        features_array = scaler.transform(features_df)

    labels = df[label_column]

    if pd.api.types.is_numeric_dtype(labels):
        targets = np.asarray(labels, dtype=np.int64)
    else:
        if label_mapping is None:
            unique_labels = sorted({str(item) for item in labels.tolist()})
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        unknown = sorted({str(item) for item in labels.tolist()} - set(label_mapping.keys()))
        if unknown:
            raise KeyError(f"Unknown labels in dataset split: {unknown}")

        targets = np.asarray([label_mapping[str(item)] for item in labels.tolist()], dtype=np.int64)

    num_classes = int(cfg.dataset.get("num_classes", int(targets.max()) + 1))
    if num_classes <= 0:
        num_classes = int(targets.max()) + 1

    return features_array.astype(np.float32), targets, num_classes, reference_columns, scaler, label_mapping
'''

new_build = r'''
def _read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_optional_test_file(cfg) -> Path | None:
    test_file = cfg.dataset.get("test_file", None)
    if not test_file:
        return None

    data_root = Path(str(cfg.dataset.get("data_root", "data")))
    candidate = Path(str(test_file))
    if not candidate.is_absolute():
        candidate = data_root / candidate

    if not candidate.exists():
        raise FileNotFoundError(f"Configured test_file does not exist: {candidate}")

    return candidate


def build_tabular_clients(cfg, device: torch.device | None) -> tuple[list[FLClient], DatasetInfo]:
    dataset_name = str(cfg.dataset.name).lower()
    data_path = _resolve_data_file(cfg, dataset_name)
    test_path = _resolve_optional_test_file(cfg)

    seed = int(cfg.runtime.get("seed", 42))
    partition = str(cfg.dataset.get("partition", "manual")).lower()
    if partition == "manual":
        partition = "label_group"

    if test_path is not None:
        train_df = _read_tabular_file(data_path)
        test_df = _read_tabular_file(test_path)

        features_train, targets_train, num_classes, reference_columns, scaler, label_mapping = _prepare_features(
            train_df,
            cfg,
            fit=True,
        )
        features_test, targets_test, _, _, _, _ = _prepare_features(
            test_df,
            cfg,
            reference_columns=reference_columns,
            scaler=scaler,
            label_mapping=label_mapping,
            fit=False,
        )

        split_source = "explicit_train_test_files"

    else:
        df = _read_tabular_file(data_path)

        features_all, targets_all, num_classes, reference_columns, scaler, label_mapping = _prepare_features(
            df,
            cfg,
            fit=True,
        )

        test_size = float(cfg.dataset.get("test_ratio", 0.2))
        if test_size <= 0.0:
            train_idx = np.arange(len(targets_all))
            test_idx = np.arange(len(targets_all))
        else:
            split_state = None if len(np.unique(targets_all)) <= 1 else targets_all
            train_idx, test_idx = train_test_split(
                np.arange(len(targets_all)),
                test_size=test_size,
                random_state=seed,
                stratify=split_state,
            )

        features_train = features_all[train_idx]
        targets_train = targets_all[train_idx]
        features_test = features_all[test_idx]
        targets_test = targets_all[test_idx]

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        split_source = f"single_file_train_test_split_{test_size}"

    print()
    print("[NIDS] Dataset preprocessing summary")
    print(f"  dataset: {dataset_name}")
    print(f"  split_source: {split_source}")
    print(f"  train_file: {data_path}")
    if test_path is not None:
        print(f"  test_file: {test_path}")
    print(f"  train shape: {train_df.shape}")
    print(f"  test shape: {test_df.shape}")
    print(f"  input_dim: {int(features_train.shape[1])}")
    print(f"  num_classes: {int(num_classes)}")
    if label_mapping is not None:
        print(f"  label_map: {label_mapping}")

    train_dataset = TabularDataset(features_train, targets_train)
    test_dataset = TabularDataset(features_test, targets_test)

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
                "split_source": split_source,
                "train_file": str(data_path),
                "test_file": str(test_path) if test_path is not None else None,
                "num_clients": int(cfg.dataset.num_clients),
                "train_samples_per_client": int(cfg.dataset.train_samples_per_client),
                "test_samples_per_client": int(cfg.dataset.test_samples_per_client),
                "val_ratio": float(cfg.dataset.val_ratio),
                "num_classes": int(num_classes),
                "input_dim": int(features_train.shape[1]),
                "true_groups": true_groups,
                "major_ratio": float(cfg.dataset.get("major_ratio", 0.85)),
                "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
                "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
                "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
                "drop_columns": list(cfg.dataset.get("drop_columns", [])),
                "auto_drop_if_contains": list(cfg.dataset.get("auto_drop_if_contains", [])),
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
            "input_dim": int(features_train.shape[1]),
            "num_true_clusters": int(cfg.dataset.get("num_true_clusters", len(true_groups))),
            "dir_alpha_inter": float(cfg.dataset.get("dir_alpha_inter", 0.1)),
            "dir_alpha_intra": float(cfg.dataset.get("dir_alpha_intra", 10.0)),
        },
    )
'''

pattern_prepare = r"def _prepare_features\(.*?(?=\ndef _default_true_groups\()"
s, n_prepare = re.subn(pattern_prepare, new_prepare.strip() + "\n\n", s, flags=re.S)

if n_prepare != 1:
    raise SystemExit(f"Expected to replace _prepare_features once, replaced {n_prepare}")

pattern_build = r"def build_tabular_clients\(.*\Z"
s, n_build = re.subn(pattern_build, new_build.strip() + "\n", s, flags=re.S)

if n_build != 1:
    raise SystemExit(f"Expected to replace build_tabular_clients once, replaced {n_build}")

p.write_text(s, encoding="utf-8")
print("patched", p)
print("replaced _prepare_features:", n_prepare)
print("replaced build_tabular_clients:", n_build)
