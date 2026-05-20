"""Metric helpers for FL evaluation."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def purity_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    true = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)
    if true.size == 0:
        return float("nan")
    total = 0
    for cluster in np.unique(pred):
        members = true[pred == cluster]
        if members.size:
            total += Counter(members.tolist()).most_common(1)[0][1]
    return float(total / true.size)


def classification_summary(client_results: list[dict[str, object]], num_classes: int) -> dict[str, float]:
    if not client_results:
        return {}
    accs = np.asarray([float(r["acc"]) for r in client_results], dtype=np.float64)
    losses = np.asarray([float(r["loss"]) for r in client_results], dtype=np.float64)
    ns = np.asarray([int(r["num_samples"]) for r in client_results], dtype=np.int64)
    y_true = np.concatenate([r["y_true"] for r in client_results]) if client_results else np.asarray([], dtype=int)
    y_pred = np.concatenate([r["y_pred"] for r in client_results]) if client_results else np.asarray([], dtype=int)
    total_correct = int((y_true == y_pred).sum()) if y_true.size else 0
    total_n = int(y_true.size)
    labels = list(range(int(num_classes)))
    return {
        "client_avg_loss": float(np.mean(losses)),
        "sample_avg_loss": float(np.average(losses, weights=np.maximum(ns, 1))),
        "client_avg_acc": float(np.mean(accs)),
        "micro_acc": float(total_correct / max(total_n, 1)),
        "global_macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "client_avg_macro_f1": float(np.mean([
            f1_score(r["y_true"], r["y_pred"], average="macro", labels=labels, zero_division=0)
            for r in client_results
        ])),
        "worst_client_acc": float(np.min(accs)),
        "best_client_acc": float(np.max(accs)),
        "acc_std": float(np.std(accs)),
        "num_eval_samples": float(total_n),
    }


def clustering_summary(true_groups: list[int | None], assignments: list[int] | np.ndarray) -> dict[str, float]:
    mask = [g is not None for g in true_groups]
    if not any(mask):
        return {"ari": float("nan"), "nmi": float("nan"), "purity": float("nan")}
    y_true = np.asarray([int(g) for g, m in zip(true_groups, mask) if m], dtype=np.int64)
    y_pred = np.asarray([int(a) for a, m in zip(assignments, mask) if m], dtype=np.int64)
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "purity": float(purity_score(y_true, y_pred)),
    }
