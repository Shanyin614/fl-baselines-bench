"""CFL / clustered federated learning baseline."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from src.flbench.core.evaluation import evaluate_cluster_models, evaluate_global_model
from src.flbench.core.sampling import sample_client_ids
from src.flbench.methods.base import BaseRunner
from src.flbench.utils.state_dict import (
    StateDict,
    apply_delta,
    clone_state,
    flatten_delta,
    weighted_average_deltas,
)


def _delta_norm(delta: StateDict) -> float:
    return float(torch.norm(flatten_delta(delta)).item())


def _cosine_similarity_matrix(deltas: list[StateDict]) -> np.ndarray:
    vecs = [flatten_delta(delta) for delta in deltas]
    n = len(vecs)
    sim = np.eye(n, dtype=np.float64)

    for i in range(n):
        vi = vecs[i]
        ni = torch.norm(vi)
        for j in range(i + 1, n):
            vj = vecs[j]
            denom = ni * torch.norm(vj)
            if float(denom.item()) <= 1e-12:
                value = 0.0
            else:
                value = float(torch.sum(vi * vj).div(denom).item())
            sim[i, j] = value
            sim[j, i] = value

    return sim


def _binary_agglomerative_from_cosine(
    deltas: list[StateDict],
    linkage: str = "complete",
) -> np.ndarray:
    sim = _cosine_similarity_matrix(deltas)
    distance = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)

    try:
        model = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=2,
            affinity="precomputed",
            linkage=linkage,
        )

    return model.fit_predict(distance)


class CFLRunner(BaseRunner):
    """Clustered Federated Learning with recursive update-similarity splits."""

    name = "cfl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.clusters: dict[int, list[int]] = {0: list(range(self.num_clients))}
        self.center_states: dict[int, StateDict] = {}

        self.next_cluster_id = 1
        self.cluster_split_events = 0
        self.split_probe_downloads = 0
        self.split_probe_uploads = 0

    def run(self) -> dict:
        total_rounds = int(self.cfg.train.total_rounds)
        warmup_rounds = int(
            self.cfg.method.get(
                "warmup_rounds",
                self.cfg.train.get("warmup_rounds", 0),
            )
        )
        warmup_rounds = min(max(warmup_rounds, 0), total_rounds)
        eval_every = int(self.cfg.eval.get("eval_every", 1))

        global_state = self.init_model_state()

        for r in range(1, warmup_rounds + 1):
            selected = sample_client_ids(
                self.num_clients,
                float(self.cfg.train.client_frac),
                self.rng,
            )
            global_state = self._fedavg_step(global_state, selected)

            if r % eval_every == 0 or r == warmup_rounds:
                metrics = evaluate_global_model(
                    clients=self.clients,
                    model_state=global_state,
                    model_fn=self.model_fn,
                    num_classes=self.num_classes,
                    split=str(self.cfg.eval.get("test_split", "test")),
                )
                self.log_metrics(
                    r,
                    "warmup",
                    metrics,
                    extra={
                        "k_config": None,
                        "num_selected_clients": len(selected),
                        "cluster_sizes": [self.num_clients],
                        "num_empty_clusters": 0,
                        "cluster_split_events": self.cluster_split_events,
                        "split_probe_downloads": self.split_probe_downloads,
                        "split_probe_uploads": self.split_probe_uploads,
                        "eps_1": float(self.cfg.method.get("eps_1", 0.4)),
                        "eps_2": float(self.cfg.method.get("eps_2", 1.6)),
                    },
                )

        self.center_states = {0: clone_state(global_state)}
        self.clusters = {0: list(range(self.num_clients))}

        if warmup_rounds == total_rounds:
            return self.save_and_summarize()

        for r in range(warmup_rounds + 1, total_rounds + 1):
            if self._should_try_split(r, warmup_rounds):
                self._maybe_split_all_clusters(r)

            selected = sample_client_ids(
                self.num_clients,
                float(self.cfg.train.client_frac),
                self.rng,
            )
            selected_by_cluster = self._group_selected_clients(selected)
            updated_clusters = self._cluster_fedavg_step(selected_by_cluster)

            if r % eval_every == 0 or r == total_rounds:
                center_states, assignments = self._dense_eval_state()
                metrics = evaluate_cluster_models(
                    self.clients,
                    center_states,
                    assignments,
                    self.model_fn,
                    self.num_classes,
                    split=str(self.cfg.eval.get("test_split", "test")),
                )
                extra = self._extra_eval_fields(num_selected=len(selected))
                extra["updated_clusters"] = updated_clusters
                self.log_metrics(r, "clustered", metrics, extra=extra)

        return self.save_and_summarize()

    def _fedavg_step(self, state: StateDict, selected: list[int]) -> StateDict:
        deltas = []
        weights = []

        for idx in selected:
            result = self.clients[idx].train(
                model_state=state,
                model_fn=self.model_fn,
                epochs=int(self.cfg.train.local_epochs),
                lr=float(self.cfg.train.lr),
                optimizer_name=str(self.cfg.train.get("optimizer", "sgd")),
                momentum=float(self.cfg.train.get("momentum", 0.9)),
                weight_decay=float(self.cfg.train.get("weight_decay", 0.0)),
                proximal_mu=float(self.cfg.train.get("proximal_mu", 0.0)),
            )
            deltas.append(result.delta)
            weights.append(result.num_samples)

        self.model_downloads += len(selected)
        self.model_uploads += len(selected)

        if not deltas:
            return state

        return apply_delta(state, weighted_average_deltas(deltas, weights))

    def _cluster_fedavg_step(self, selected_by_cluster: dict[int, list[int]]) -> list[int]:
        updated_clusters: list[int] = []

        for cluster_id, selected in selected_by_cluster.items():
            if not selected:
                continue

            base_state = self.center_states[cluster_id]
            deltas = []
            weights = []

            for idx in selected:
                result = self.clients[idx].train(
                    model_state=base_state,
                    model_fn=self.model_fn,
                    epochs=int(self.cfg.train.local_epochs),
                    lr=float(self.cfg.train.lr),
                    optimizer_name=str(self.cfg.train.get("optimizer", "sgd")),
                    momentum=float(self.cfg.train.get("momentum", 0.9)),
                    weight_decay=float(self.cfg.train.get("weight_decay", 0.0)),
                    proximal_mu=float(self.cfg.train.get("proximal_mu", 0.0)),
                )
                deltas.append(result.delta)
                weights.append(result.num_samples)

            self.model_downloads += len(selected)
            self.model_uploads += len(selected)

            if deltas:
                self.center_states[cluster_id] = apply_delta(
                    base_state,
                    weighted_average_deltas(deltas, weights),
                )
                updated_clusters.append(int(cluster_id))

        return updated_clusters

    def _should_try_split(self, round_idx: int, warmup_rounds: int) -> bool:
        min_split_round = int(self.cfg.method.get("min_split_round", warmup_rounds + 1))
        interval = int(self.cfg.method.get("clustering_interval", 1))

        if interval <= 0:
            return False
        if round_idx < min_split_round:
            return False

        return (round_idx - warmup_rounds) % interval == 0

    def _maybe_split_all_clusters(self, round_idx: int) -> None:
        for cluster_id in list(sorted(self.clusters.keys())):
            if cluster_id in self.clusters:
                self._maybe_split_cluster(cluster_id, round_idx)

    def _maybe_split_cluster(self, cluster_id: int, round_idx: int) -> bool:
        members = list(self.clusters[cluster_id])
        min_cluster_size = int(self.cfg.method.get("min_cluster_size", 4))

        if len(members) < 2 * min_cluster_size:
            return False

        base_state = self.center_states[cluster_id]
        deltas = []
        weights = []

        for idx in members:
            result = self.clients[idx].train(
                model_state=base_state,
                model_fn=self.model_fn,
                epochs=int(self.cfg.method.get("split_probe_epochs", 1)),
                lr=float(self.cfg.train.lr),
                optimizer_name=str(self.cfg.train.get("optimizer", "sgd")),
                momentum=float(self.cfg.train.get("momentum", 0.9)),
                weight_decay=float(self.cfg.train.get("weight_decay", 0.0)),
                proximal_mu=float(self.cfg.train.get("proximal_mu", 0.0)),
            )
            deltas.append(result.delta)
            weights.append(result.num_samples)

        self.model_downloads += len(members)
        self.model_uploads += len(members)
        self.split_probe_downloads += len(members)
        self.split_probe_uploads += len(members)

        mean_delta = weighted_average_deltas(deltas, [1.0 for _ in deltas])
        mean_norm = _delta_norm(mean_delta)
        max_norm = max(_delta_norm(delta) for delta in deltas)

        eps_1 = float(self.cfg.method.get("eps_1", 0.4))
        eps_2 = float(self.cfg.method.get("eps_2", 1.6))

        if not (mean_norm < eps_1 and max_norm > eps_2):
            print(
                f"[cfl] round={round_idx} cluster={cluster_id} no split "
                f"size={len(members)} mean_norm={mean_norm:.4f} "
                f"max_norm={max_norm:.4f}"
            )
            return False

        linkage = str(self.cfg.method.get("linkage", "complete"))
        labels = _binary_agglomerative_from_cosine(deltas, linkage=linkage)

        left = [members[i] for i, label in enumerate(labels) if int(label) == 0]
        right = [members[i] for i, label in enumerate(labels) if int(label) == 1]

        if len(left) < min_cluster_size or len(right) < min_cluster_size:
            print(
                f"[cfl] round={round_idx} cluster={cluster_id} rejected split "
                f"sizes=({len(left)}, {len(right)})"
            )
            return False

        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1

        self.clusters[cluster_id] = left
        self.clusters[new_cluster_id] = right

        split_center_init = str(self.cfg.method.get("split_center_init", "parent"))
        if split_center_init == "probe_avg":
            left_deltas = [deltas[i] for i, label in enumerate(labels) if int(label) == 0]
            left_weights = [weights[i] for i, label in enumerate(labels) if int(label) == 0]
            right_deltas = [deltas[i] for i, label in enumerate(labels) if int(label) == 1]
            right_weights = [weights[i] for i, label in enumerate(labels) if int(label) == 1]

            self.center_states[cluster_id] = apply_delta(
                base_state,
                weighted_average_deltas(left_deltas, left_weights),
            )
            self.center_states[new_cluster_id] = apply_delta(
                base_state,
                weighted_average_deltas(right_deltas, right_weights),
            )
        elif split_center_init == "parent":
            self.center_states[cluster_id] = clone_state(base_state)
            self.center_states[new_cluster_id] = clone_state(base_state)
        else:
            raise ValueError(f"unknown CFL split_center_init: {split_center_init}")

        self.cluster_split_events += 1

        print(
            f"[cfl] round={round_idx} split cluster={cluster_id} "
            f"-> {cluster_id}:{len(left)}, {new_cluster_id}:{len(right)} "
            f"mean_norm={mean_norm:.4f} max_norm={max_norm:.4f}"
        )

        return True

    def _group_selected_clients(self, selected: list[int]) -> dict[int, list[int]]:
        grouped = {cluster_id: [] for cluster_id in self.clusters}

        client_to_cluster = {}
        for cluster_id, members in self.clusters.items():
            for idx in members:
                client_to_cluster[int(idx)] = int(cluster_id)

        for idx in selected:
            grouped[client_to_cluster[int(idx)]].append(int(idx))

        return grouped

    def _dense_eval_state(self) -> tuple[list[StateDict], np.ndarray]:
        cluster_ids = sorted(self.clusters.keys())
        cluster_to_dense = {cluster_id: pos for pos, cluster_id in enumerate(cluster_ids)}

        center_states = [self.center_states[cluster_id] for cluster_id in cluster_ids]
        assignments = np.zeros(self.num_clients, dtype=np.int64)

        for cluster_id, members in self.clusters.items():
            dense_id = cluster_to_dense[cluster_id]
            for idx in members:
                assignments[int(idx)] = int(dense_id)

        return center_states, assignments

    def _cluster_sizes(self) -> list[int]:
        return [len(self.clusters[cluster_id]) for cluster_id in sorted(self.clusters.keys())]

    def _extra_eval_fields(self, num_selected: int) -> dict:
        return {
            "k_config": None,
            "num_selected_clients": int(num_selected),
            "cluster_sizes": self._cluster_sizes(),
            "num_empty_clusters": 0,
            "cluster_split_events": int(self.cluster_split_events),
            "split_probe_downloads": int(self.split_probe_downloads),
            "split_probe_uploads": int(self.split_probe_uploads),
            "eps_1": float(self.cfg.method.get("eps_1", 0.4)),
            "eps_2": float(self.cfg.method.get("eps_2", 1.6)),
            "clustering_interval": int(self.cfg.method.get("clustering_interval", 1)),
            "min_cluster_size": int(self.cfg.method.get("min_cluster_size", 4)),
            "split_center_init": str(self.cfg.method.get("split_center_init", "parent")),
        }
