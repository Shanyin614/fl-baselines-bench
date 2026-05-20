from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from flbench.methods.base import BaseMethod


StateDict = Dict[str, torch.Tensor]


def _clone_state(state: StateDict) -> StateDict:
    return {k: v.detach().clone() for k, v in state.items()}

def _weighted_average(states: List[StateDict], weights: List[int]) -> StateDict:
    """
    Sample-weighted FedAvg for model state_dicts.
    Floating tensors are weighted averaged.
    Non-floating buffers are copied from the first client.
    """
    if len(states) == 0:
        raise ValueError("Cannot average empty state list.")

    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length.")

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("Total aggregation weight must be positive.")

    avg_state: StateDict = {}

    for key in states[0].keys():
        first_tensor = states[0][key].detach().cpu()

        if torch.is_floating_point(first_tensor):
            acc = torch.zeros_like(first_tensor, dtype=torch.float32)

            for state, weight in zip(states, weights):
                tensor = state[key].detach().cpu().float()
                acc += tensor * (float(weight) / total_weight)

            avg_state[key] = acc.to(dtype=first_tensor.dtype)
        else:
            avg_state[key] = first_tensor.clone()

    return avg_state

def _state_delta(new_state: StateDict, old_state: StateDict) -> StateDict:
    return {
        k: (new_state[k].detach().cpu() - old_state[k].detach().cpu())
        for k in old_state.keys()
    }


def _add_delta(base_state: StateDict, delta: StateDict) -> StateDict:
    return {
        k: (base_state[k].detach().cpu() + delta[k].detach().cpu())
        for k in base_state.keys()
    }


def _flatten_delta(delta: StateDict) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float().cpu() for v in delta.values()])


def _mean_delta(deltas: List[StateDict]) -> StateDict:
    if len(deltas) == 0:
        raise ValueError("Cannot average an empty delta list.")

    out = {}
    for k in deltas[0].keys():
        out[k] = torch.stack([d[k].float().cpu() for d in deltas], dim=0).mean(dim=0)
    return out


def _cosine_similarity_matrix(deltas: List[StateDict]) -> np.ndarray:
    vecs = [_flatten_delta(d) for d in deltas]
    n = len(vecs)
    sim = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            denom = torch.norm(vecs[i]) * torch.norm(vecs[j]) + 1e-12
            value = torch.sum(vecs[i] * vecs[j]) / denom
            value = float(value.item())
            sim[i, j] = value
            sim[j, i] = value

    return sim


def _update_norm(delta: StateDict) -> float:
    return float(torch.norm(_flatten_delta(delta)).item())


def _fit_binary_agglomerative(distance: np.ndarray) -> np.ndarray:
    """
    sklearn changed `affinity` to `metric` in newer versions.
    This helper supports both.
    """
    try:
        model = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="complete",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=2,
            affinity="precomputed",
            linkage="complete",
        )

    return model.fit_predict(distance)


class CFLRunner(BaseMethod):
    """
    Clustered Federated Learning baseline.

    This implementation intentionally keeps CARES out of the repo.
    It is a baseline-only method compatible with the same protocol/split/model
    used by CARES and FeSEM.
    """

    name = "cfl"

    def __init__(self, clients, model_fn, cfg, logger):
        super().__init__(clients, model_fn, cfg, logger)

        self.cfg = cfg
        self.clients = clients
        self.model_fn = model_fn
        self.logger = logger

        self.method_cfg = cfg.method
        self.train_cfg = cfg.train

        self.total_rounds = int(cfg.train.total_rounds)
        self.warmup_rounds = int(
            getattr(self.method_cfg, "warmup_rounds", getattr(cfg.train, "warmup_rounds", 0))
        )

        self.eps_1 = float(getattr(self.method_cfg, "eps_1", 0.4))
        self.eps_2 = float(getattr(self.method_cfg, "eps_2", 1.6))
        self.clustering_interval = int(getattr(self.method_cfg, "clustering_interval", 1))
        self.min_cluster_size = int(getattr(self.method_cfg, "min_cluster_size", 4))
        self.min_split_round = int(getattr(self.method_cfg, "min_split_round", self.warmup_rounds + 1))
        self.split_center_init = str(getattr(self.method_cfg, "split_center_init", "parent"))

        self.seed = int(getattr(cfg.runtime, "seed", 42))
        self.device = torch.device(getattr(cfg.runtime, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        # cluster_id -> list[client_id]
        self.clusters: Dict[int, List[int]] = {
            0: [client.client_id for client in self.clients]
        }

        # cluster_id -> model state
        self.cluster_states: Dict[int, StateDict] = {}

        self.next_cluster_id = 1

        # accounting
        self.model_downloads = 0
        self.model_uploads = 0
        self.split_probe_downloads = 0
        self.split_probe_uploads = 0
        self.cluster_split_events = 0

    # ---------------------------------------------------------------------
    # Basic helpers
    # ---------------------------------------------------------------------

    def _new_model_state(self) -> StateDict:
        model = self.model_fn().to(self.device)
        return _clone_state(model.state_dict())

    def _client_by_id(self, client_id: int):
        return self.clients[int(client_id)]

    def _sample_client_ids(self, client_ids: List[int], round_idx: int, cluster_id: int) -> List[int]:
        if len(client_ids) == 0:
            return []

        frac = float(self.train_cfg.client_frac)
        n_selected = max(1, int(math.ceil(len(client_ids) * frac)))

        rng = np.random.default_rng(self.seed + round_idx * 1009 + cluster_id * 9173)
        selected = rng.choice(client_ids, size=min(n_selected, len(client_ids)), replace=False)
        return [int(x) for x in selected.tolist()]

    def _train_clients_from_state(
        self,
        client_ids: List[int],
        base_state: StateDict,
    ) -> Tuple[List[StateDict], List[int], List[StateDict]]:
        updated_states = []
        weights = []
        deltas = []

        for client_id in client_ids:
            client = self._client_by_id(client_id)

            new_state, num_samples = client.train(
                model_state=base_state,
                epochs=int(self.train_cfg.local_epochs),
                lr=float(self.train_cfg.lr),
            )

            updated_states.append(new_state)
            weights.append(int(num_samples))
            deltas.append(_state_delta(new_state, base_state))

        return updated_states, weights, deltas

    # ---------------------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------------------

    def _fedavg_warmup(self):
        global_state = self._new_model_state()

        for round_idx in range(1, self.warmup_rounds + 1):
            all_client_ids = [client.client_id for client in self.clients]
            selected_ids = self._sample_client_ids(all_client_ids, round_idx, cluster_id=0)

            updated_states, weights, _ = self._train_clients_from_state(selected_ids, global_state)

            self.model_downloads += len(selected_ids)
            self.model_uploads += len(selected_ids)

            if len(updated_states) > 0:
                global_state = _weighted_average(updated_states, weights)

            self.cluster_states = {0: _clone_state(global_state)}
            self.clusters = {0: all_client_ids}

            metrics = self.evaluate(round_idx=round_idx, phase="warmup")
            self._log_round(round_idx, "warmup", metrics, updated_clusters=[0])

        self.cluster_states = {0: _clone_state(global_state)}
        self.clusters = {0: [client.client_id for client in self.clients]}

    # ---------------------------------------------------------------------
    # CFL split
    # ---------------------------------------------------------------------

    def _should_try_split(self, round_idx: int) -> bool:
        if round_idx < self.min_split_round:
            return False
        if self.clustering_interval <= 0:
            return False
        return (round_idx - self.warmup_rounds) % self.clustering_interval == 0

    def _maybe_split_cluster(self, cluster_id: int, round_idx: int) -> bool:
        client_ids = self.clusters[cluster_id]

        if len(client_ids) < self.min_cluster_size:
            return False

        base_state = self.cluster_states[cluster_id]

        # CFL split probe: all clients in this cluster compute updates from the same parent model.
        _, _, deltas = self._train_clients_from_state(client_ids, base_state)

        self.split_probe_downloads += len(client_ids)
        self.split_probe_uploads += len(client_ids)

        mean_update = _mean_delta(deltas)
        mean_norm = _update_norm(mean_update)
        max_norm = max(_update_norm(d) for d in deltas)

        if not (mean_norm < self.eps_1 and max_norm > self.eps_2):
            print(
                f"[cfl] round={round_idx} cluster={cluster_id} no split "
                f"mean_norm={mean_norm:.4f} max_norm={max_norm:.4f} size={len(client_ids)}"
            )
            return False

        sim = _cosine_similarity_matrix(deltas)
        distance = 1.0 - sim

        labels = _fit_binary_agglomerative(distance)

        left = [client_ids[i] for i in range(len(client_ids)) if labels[i] == 0]
        right = [client_ids[i] for i in range(len(client_ids)) if labels[i] == 1]

        if len(left) == 0 or len(right) == 0:
            return False

        if len(left) < self.min_cluster_size // 2 or len(right) < self.min_cluster_size // 2:
            return False

        old_cluster_id = cluster_id
        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1

        self.clusters[old_cluster_id] = left
        self.clusters[new_cluster_id] = right

        if self.split_center_init == "probe_avg":
            left_deltas = [deltas[i] for i in range(len(client_ids)) if labels[i] == 0]
            right_deltas = [deltas[i] for i in range(len(client_ids)) if labels[i] == 1]

            self.cluster_states[old_cluster_id] = _add_delta(base_state, _mean_delta(left_deltas))
            self.cluster_states[new_cluster_id] = _add_delta(base_state, _mean_delta(right_deltas))
        else:
            self.cluster_states[old_cluster_id] = _clone_state(base_state)
            self.cluster_states[new_cluster_id] = _clone_state(base_state)

        self.cluster_split_events += 1

        print(
            f"[cfl] round={round_idx} split cluster={cluster_id} "
            f"-> sizes=({len(left)}, {len(right)}) "
            f"mean_norm={mean_norm:.4f} max_norm={max_norm:.4f}"
        )

        return True

    def _maybe_split_all_clusters(self, round_idx: int):
        if not self._should_try_split(round_idx):
            return

        # freeze ids because splitting mutates self.clusters
        cluster_ids = list(self.clusters.keys())

        for cluster_id in cluster_ids:
            if cluster_id in self.clusters:
                self._maybe_split_cluster(cluster_id, round_idx)

    # ---------------------------------------------------------------------
    # Cluster-wise FedAvg
    # ---------------------------------------------------------------------

    def _train_one_cluster(self, cluster_id: int, round_idx: int) -> bool:
        client_ids = self.clusters[cluster_id]
        if len(client_ids) == 0:
            return False

        selected_ids = self._sample_client_ids(client_ids, round_idx, cluster_id)
        base_state = self.cluster_states[cluster_id]

        updated_states, weights, _ = self._train_clients_from_state(selected_ids, base_state)

        self.model_downloads += len(selected_ids)
        self.model_uploads += len(selected_ids)

        if len(updated_states) > 0:
            self.cluster_states[cluster_id] = _weighted_average(updated_states, weights)
            return True

        return False

    def _train_clusterwise_round(self, round_idx: int) -> List[int]:
        updated_clusters = []

        for cluster_id in list(self.clusters.keys()):
            updated = self._train_one_cluster(cluster_id, round_idx)
            if updated:
                updated_clusters.append(cluster_id)

        return updated_clusters

    # ---------------------------------------------------------------------
    # Evaluation/logging hooks
    # ---------------------------------------------------------------------

    def _client_assignments(self) -> Dict[int, int]:
        assignments = {}
        for cluster_id, client_ids in self.clusters.items():
            for client_id in client_ids:
                assignments[int(client_id)] = int(cluster_id)
        return assignments

    def _cluster_sizes(self) -> List[int]:
        return [len(v) for _, v in sorted(self.clusters.items(), key=lambda x: x[0])]

    def evaluate(self, round_idx: int, phase: str) -> dict:
        """
        This calls the common evaluator expected by the existing benchmark.

        If your BaseMethod already has evaluate_clustered/evaluate_personalized,
        replace this function body with the same evaluator call used in FeSEM.
        """
        assignments = self._client_assignments()

        # Most likely existing helper in your BaseMethod from FeSEM.
        if hasattr(super(), "evaluate"):
            return super().evaluate(
                round_idx=round_idx,
                phase=phase,
                assignments=assignments,
                center_states=self.cluster_states,
            )

        raise RuntimeError(
            "CFLRunner.evaluate needs to be wired to the same evaluation helper used by FeSEM. "
            "Copy FeSEMRunner's evaluation call here and pass assignments + cluster_states."
        )

    def _log_round(self, round_idx: int, phase: str, metrics: dict, updated_clusters: List[int]):
        assignments = self._client_assignments()
        cluster_sizes = self._cluster_sizes()

        row = dict(metrics)
        row.update(
            {
                "method": "cfl",
                "round": round_idx,
                "phase": phase,
                "k_config": None,
                "k_pred": len([s for s in cluster_sizes if s > 0]),
                "cluster_sizes": cluster_sizes,
                "num_empty_clusters": sum(1 for s in cluster_sizes if s == 0),
                "model_downloads": self.model_downloads,
                "model_uploads": self.model_uploads,
                "split_probe_downloads": self.split_probe_downloads,
                "split_probe_uploads": self.split_probe_uploads,
                "model_transmissions": (
                    self.model_downloads
                    + self.model_uploads
                    + self.split_probe_downloads
                    + self.split_probe_uploads
                ),
                "cluster_split_events": self.cluster_split_events,
                "eps_1": self.eps_1,
                "eps_2": self.eps_2,
                "updated_clusters": updated_clusters,
                "assignment_interval": self.clustering_interval,
            }
        )

        self.logger.log_round(row)

        print(
            f"[eval] round={round_idx} phase={phase} "
            f"client_avg_acc={row.get('client_avg_acc', float('nan')):.4f} "
            f"micro_acc={row.get('micro_acc', float('nan')):.4f} "
            f"k_pred={row['k_pred']}"
        )

    # ---------------------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------------------

    def run(self):
        start = time.time()

        self._fedavg_warmup()

        for round_idx in range(self.warmup_rounds + 1, self.total_rounds + 1):
            self._maybe_split_all_clusters(round_idx)

            updated_clusters = self._train_clusterwise_round(round_idx)

            metrics = self.evaluate(round_idx=round_idx, phase="clustered")
            self._log_round(round_idx, "clustered", metrics, updated_clusters)

        final_summary = self.logger.rows[-1].copy()
        final_summary["runtime_sec"] = time.time() - start

        output_path = self.logger.save(final_summary)
        final_summary["output_path"] = output_path

        return final_summary
