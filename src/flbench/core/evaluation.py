"""Evaluation routines shared by methods."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
from torch import nn

from flbench.core.client import FLClient
from flbench.core.metrics import classification_summary, clustering_summary
from flbench.utils.state_dict import StateDict


def evaluate_global_model(
    clients: list[FLClient],
    model_state: StateDict,
    model_fn: Callable[[], nn.Module],
    num_classes: int,
    split: str = "test",
) -> dict[str, float]:
    results = [client.evaluate_state(model_state, model_fn, split=split) for client in clients]
    metrics = classification_summary(results, num_classes=num_classes)
    assignments = [0 for _ in clients]
    true_groups = [client.group_id for client in clients]
    metrics.update(clustering_summary(true_groups, assignments))
    metrics["k_pred"] = 1.0
    return metrics


def evaluate_cluster_models(
    clients: list[FLClient],
    center_states: list[StateDict],
    assignments: list[int] | np.ndarray,
    model_fn: Callable[[], nn.Module],
    num_classes: int,
    split: str = "test",
) -> dict[str, float]:
    assignments_arr = np.asarray(assignments, dtype=np.int64)
    results = []
    for idx, client in enumerate(clients):
        k = int(assignments_arr[idx])
        results.append(client.evaluate_state(center_states[k], model_fn, split=split))
    metrics = classification_summary(results, num_classes=num_classes)
    true_groups = [client.group_id for client in clients]
    metrics.update(clustering_summary(true_groups, assignments_arr.tolist()))
    metrics["k_pred"] = float(len(np.unique(assignments_arr)))
    return metrics
