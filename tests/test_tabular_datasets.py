import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flbench.datasets.factory import build_clients
from flbench.models.factory import build_model_fn
from flbench.utils.config import ConfigNode


def test_build_clients_supports_tabular_dataset(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_2": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = ConfigNode(
        {
            "dataset": {
                "name": "cicids2017",
                "data_root": str(tmp_path),
                "data_file": str(csv_path),
                "label_column": "label",
                "num_clients": 6,
                "train_samples_per_client": 4,
                "test_samples_per_client": 1,
                "val_ratio": 0.2,
                "partition": "manual",
                "major_ratio": 0.85,
                "num_classes": 2,
                "num_workers": 0,
            },
            "model": {"name": "mlp", "num_classes": 2, "input_dim": 2},
            "train": {"batch_size": 2},
            "runtime": {"seed": 7},
        }
    )

    clients, dataset_info = build_clients(cfg, device=None)
    assert len(clients) == 6
    assert dataset_info.name == "cicids2017"
    assert clients[0].num_train > 0
    assert clients[0].num_test > 0

    model_fn = build_model_fn(cfg)
    model = model_fn()
    assert model is not None
