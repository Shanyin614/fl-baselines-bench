"""Base runner class."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from flbench.core.client import FLClient
from flbench.core.logging import CSVLogger
from flbench.core.types import DatasetInfo
from flbench.utils.state_dict import StateDict, clone_state


class BaseRunner:
    name = "base"

    def __init__(
        self,
        clients: list[FLClient],
        model_fn: Callable[[], nn.Module],
        cfg,
        device: torch.device,
        dataset_info: DatasetInfo,
    ) -> None:
        self.clients = clients
        self.model_fn = model_fn
        self.cfg = cfg
        self.device = device
        self.dataset_info = dataset_info
        self.seed = int(cfg.runtime.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)
        self.num_clients = len(clients)
        self.num_classes = int(dataset_info.num_classes)
        self.start_time = time.time()
        output_dir = str(cfg.output.get("dir", "outputs/csv"))
        output_name = str(cfg.output.get("name", f"{self.name}_seed{self.seed}.csv"))
        self.logger = CSVLogger(output_dir, output_name)
        self.model_downloads = 0
        self.model_uploads = 0
        self.assignment_model_evals = 0

    def init_model_state(self) -> StateDict:
        model = self.model_fn()
        return clone_state(model.state_dict())

    @property
    def model_transmissions(self) -> int:
        return int(self.model_downloads + self.model_uploads + self.assignment_model_evals)

    def _base_row(self, round_idx: int, phase: str) -> dict:
        return {
            "method": self.name,
            "dataset": self.dataset_info.name,
            "seed": self.seed,
            "round": int(round_idx),
            "phase": phase,
            "total_rounds": int(self.cfg.train.total_rounds),
            "client_frac": float(self.cfg.train.client_frac),
            "local_epochs": int(self.cfg.train.local_epochs),
            "lr": float(self.cfg.train.lr),
            "batch_size": int(self.cfg.train.batch_size),
            "num_clients": int(self.num_clients),
            "k_config": int(self.cfg.method.get("num_clusters", 1)) if "method" in self.cfg else 1,
            "model_downloads": int(self.model_downloads),
            "model_uploads": int(self.model_uploads),
            "assignment_model_evals": int(self.assignment_model_evals),
            "model_transmissions": int(self.model_transmissions),
            "runtime_sec": float(time.time() - self.start_time),
            "split_file": self.dataset_info.split_file,
        }

    def log_metrics(self, round_idx: int, phase: str, metrics: dict, extra: dict | None = None) -> None:
        row = self._base_row(round_idx=round_idx, phase=phase)
        row.update(metrics)
        if extra:
            row.update(extra)
        self.logger.log(row)
        important = [
            f"round={round_idx}",
            f"phase={phase}",
            f"client_avg_acc={metrics.get('client_avg_acc', float('nan')):.4f}",
            f"micro_acc={metrics.get('micro_acc', float('nan')):.4f}",
        ]
        if "k_pred" in metrics:
            important.append(f"k_pred={int(metrics['k_pred'])}")
        print("[eval] " + " ".join(important))

    def save_and_summarize(self) -> dict:
        path = self.logger.save()
        print(f"[logger] Saved round metrics to {path}")
        if not self.logger.rows:
            return {"output_path": str(path)}
        last = dict(self.logger.rows[-1])
        last["output_path"] = str(path)
        return last

    def run(self) -> dict:
        raise NotImplementedError
