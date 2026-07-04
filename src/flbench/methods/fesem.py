"""FeSEM / fixed-K multi-center federated learning baseline."""
from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans

from flbench.core.evaluation import evaluate_cluster_models, evaluate_global_model
from flbench.core.sampling import sample_client_ids
from flbench.methods.base import BaseRunner
from flbench.utils.state_dict import (
    StateDict,
    apply_delta,
    flatten_delta,
    perturb_state,
    weighted_average_deltas,
    weighted_average_states,
)


class FeSEMRunner(BaseRunner):
    """Fixed-K multi-center FL with validation-loss EM assignment.

    This runner intentionally uses the same model, optimizer, local epochs,
    client sampling, and total communication budget as the CARES protocol.
    Warm-up rounds are included in `train.total_rounds`.
    """

    name = "fesem"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.K = int(self.cfg.method.num_clusters)
        if self.K < 1:
            raise ValueError("FeSEM requires method.num_clusters >= 1")
        self.center_states: list[StateDict] = []
        self.assignments = np.zeros(self.num_clients, dtype=np.int64)
        self.global_state: StateDict | None = None

    def run(self) -> dict:
        total_rounds = int(self.cfg.train.total_rounds)
        warmup_rounds = int(self.cfg.train.get("warmup_rounds", 0))
        warmup_rounds = min(max(warmup_rounds, 0), total_rounds)
        eval_every = int(self.cfg.eval.get("eval_every", 1))

        self.global_state = self.init_model_state()

        # Phase 1: FedAvg warm-up. These rounds are part of the total budget.
        for r in range(1, warmup_rounds + 1):
            selected = sample_client_ids(self.num_clients, float(self.cfg.train.client_frac), self.rng)
            self.global_state = self._fedavg_step(self.global_state, selected)
            if r % eval_every == 0 or r == warmup_rounds:
                metrics = evaluate_global_model(
                    clients=self.clients,
                    model_state=self.global_state,
                    model_fn=self.model_fn,
                    num_classes=self.num_classes,
                    split=str(self.cfg.eval.get("test_split", "test")),
                )
                self.log_metrics(r, "warmup", metrics, extra={"num_selected_clients": len(selected), "cluster_sizes": [self.num_clients]})

        # Phase 2 init: create center models and initial assignments.
        self._initialize_centers(self.global_state)
        self.assign_clients()
        self.repair_tiny_clusters()

        if warmup_rounds == total_rounds:
            metrics = evaluate_cluster_models(
                self.clients,
                self.center_states,
                self.assignments,
                self.model_fn,
                self.num_classes,
                split=str(self.cfg.eval.get("test_split", "test")),
            )
            self.log_metrics(total_rounds, "final", metrics, extra=self._extra_eval_fields(0))
            return self.save_and_summarize()

        # Phase 2: EM-style repeated assignment and cluster-wise FedAvg.
        for r in range(warmup_rounds + 1, total_rounds + 1):
            clustered_step = r - warmup_rounds
            if clustered_step == 1 or clustered_step % int(self.cfg.method.get("assignment_interval", 1)) == 0:
                self.assign_clients()
                self.repair_tiny_clusters()

            selected = sample_client_ids(self.num_clients, float(self.cfg.train.client_frac), self.rng)
            selected_by_cluster = self._group_selected_clients(selected)
            updated_clusters = self._cluster_fedavg_step(selected_by_cluster)

            if r % eval_every == 0 or r == total_rounds:
                metrics = evaluate_cluster_models(
                    self.clients,
                    self.center_states,
                    self.assignments,
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
            )
            deltas.append(result.delta)
            weights.append(result.num_samples)
        self.model_downloads += len(selected)
        self.model_uploads += len(selected)
        if not deltas:
            return state
        return apply_delta(state, weighted_average_deltas(deltas, weights))

    def _initialize_centers(self, base_state: StateDict) -> None:
        init = str(self.cfg.method.get("init", "random_perturb")).lower()
        if init == "same":
            self.center_states = [dict((k, v.clone()) for k, v in base_state.items()) for _ in range(self.K)]
        elif init == "random_perturb":
            sigma = float(self.cfg.method.get("center_perturb_sigma", 0.01))
            self.center_states = [perturb_state(base_state, sigma=sigma, seed=self.seed + 1000 + k) for k in range(self.K)]
        elif init == "warmup_kmeans":
            self.center_states = self._warmup_kmeans_init(base_state)
        else:
            raise ValueError(f"unknown FeSEM init: {init}")
        print(f"[fesem] initialized {len(self.center_states)} centers using init={init}")

    def _warmup_kmeans_init(self, base_state: StateDict) -> list[StateDict]:
        """Initialize centers by clustering one-step client updates from the warm-up model.

        This is optional because it adds an initialization cost. The cost is counted
        in model_downloads/uploads.
        """
        sample_frac = float(self.cfg.method.get("kmeans_sample_frac", 1.0))
        if sample_frac >= 1.0:
            client_ids = list(range(self.num_clients))
        else:
            client_ids = sample_client_ids(self.num_clients, sample_frac, self.rng)
        vectors = []
        new_states = []
        weights = []
        for idx in client_ids:
            result = self.clients[idx].train(
                model_state=base_state,
                model_fn=self.model_fn,
                epochs=int(self.cfg.method.get("kmeans_local_epochs", 1)),
                lr=float(self.cfg.train.lr),
                optimizer_name=str(self.cfg.train.get("optimizer", "sgd")),
                momentum=float(self.cfg.train.get("momentum", 0.9)),
                weight_decay=float(self.cfg.train.get("weight_decay", 0.0)),
            )
            vectors.append(flatten_delta(result.delta).numpy())
            new_states.append(result.new_state)
            weights.append(result.num_samples)
        self.model_downloads += len(client_ids)
        self.model_uploads += len(client_ids)

        if len(client_ids) < self.K:
            sigma = float(self.cfg.method.get("center_perturb_sigma", 0.01))
            return [perturb_state(base_state, sigma=sigma, seed=self.seed + 2000 + k) for k in range(self.K)]

        X = np.stack(vectors, axis=0)
        km = KMeans(n_clusters=self.K, n_init=10, random_state=self.seed + 2000)
        labels = km.fit_predict(X)
        centers: list[StateDict] = []
        for k in range(self.K):
            member_pos = [pos for pos, label in enumerate(labels) if int(label) == k]
            if member_pos:
                states_k = [new_states[pos] for pos in member_pos]
                weights_k = [weights[pos] for pos in member_pos]
                centers.append(weighted_average_states(states_k, weights_k))
            else:
                centers.append(perturb_state(base_state, sigma=0.01, seed=self.seed + 3000 + k))
        return centers

    def assign_clients(self) -> None:
        split = str(self.cfg.method.get("assignment_split", self.cfg.eval.get("assignment_split", "val")))
        new_assignments = np.zeros(self.num_clients, dtype=np.int64)
        for i, client in enumerate(self.clients):
            losses = [client.loss_on_model(state, self.model_fn, split=split) for state in self.center_states]
            new_assignments[i] = int(np.argmin(losses))
        # Naive implementation accounting: each client evaluates K candidate models.
        self.assignment_model_evals += self.num_clients * self.K
        self.assignments = new_assignments
        print(f"[fesem] assignment cluster sizes: {self.cluster_sizes()}")

    def repair_tiny_clusters(self) -> None:
        min_size = int(self.cfg.method.get("min_cluster_size", 1))
        if min_size <= 1:
            return
        sizes = self.cluster_sizes()
        large = [k for k, size in enumerate(sizes) if size >= min_size]
        if not large:
            # Keep the largest cluster as the anchor if all clusters are tiny.
            large = [int(np.argmax(np.asarray(sizes)))]
        changed = False
        for k, size in enumerate(sizes):
            if size == 0 or size >= min_size:
                continue
            members = np.where(self.assignments == k)[0].tolist()
            for idx in members:
                # Reassign to the best large cluster by validation loss.
                losses = [self.clients[idx].loss_on_model(self.center_states[g], self.model_fn, split="val") for g in large]
                self.assignment_model_evals += len(large)
                self.assignments[idx] = int(large[int(np.argmin(losses))])
                changed = True
        if changed:
            self._relabel_nonempty_clusters()
            print(f"[fesem] repaired tiny clusters: {self.cluster_sizes()}")
        self._handle_empty_clusters()

    def _handle_empty_clusters(self) -> None:
        sizes = self.cluster_sizes(include_empty=True)
        empty = [k for k, s in enumerate(sizes) if s == 0]
        if not empty:
            return
        strategy = str(self.cfg.method.get("empty_cluster_strategy", "keep")).lower()
        if strategy == "keep":
            return
        if strategy == "random_perturb":
            assert self.global_state is not None
            sigma = float(self.cfg.method.get("center_perturb_sigma", 0.01))
            for k in empty:
                self.center_states[k] = perturb_state(self.global_state, sigma=sigma, seed=self.seed + 4000 + k)
            return
        raise ValueError(f"unknown empty_cluster_strategy: {strategy}")

    def _relabel_nonempty_clusters(self) -> None:
        unique = sorted(np.unique(self.assignments).astype(int).tolist())
        mapping = {old: new for new, old in enumerate(unique)}
        old_centers = self.center_states
        self.assignments = np.asarray([mapping[int(a)] for a in self.assignments], dtype=np.int64)
        self.center_states = [old_centers[old] for old in unique]
        self.K = len(self.center_states)

    def _group_selected_clients(self, selected: list[int]) -> dict[int, list[int]]:
        grouped = {k: [] for k in range(len(self.center_states))}
        for idx in selected:
            grouped[int(self.assignments[idx])].append(idx)
        return grouped

    def _cluster_fedavg_step(self, selected_by_cluster: dict[int, list[int]]) -> list[int]:
        updated_clusters: list[int] = []
        for k, selected in selected_by_cluster.items():
            if not selected:
                continue
            deltas = []
            weights = []
            base_state = self.center_states[k]
            for idx in selected:
                result = self.clients[idx].train(
                    model_state=base_state,
                    model_fn=self.model_fn,
                    epochs=int(self.cfg.train.local_epochs),
                    lr=float(self.cfg.train.lr),
                    optimizer_name=str(self.cfg.train.get("optimizer", "sgd")),
                    momentum=float(self.cfg.train.get("momentum", 0.9)),
                    weight_decay=float(self.cfg.train.get("weight_decay", 0.0)),
                )
                deltas.append(result.delta)
                weights.append(result.num_samples)
            self.model_downloads += len(selected)
            self.model_uploads += len(selected)
            if deltas:
                self.center_states[k] = apply_delta(base_state, weighted_average_deltas(deltas, weights))
                updated_clusters.append(k)
        return updated_clusters

    def cluster_sizes(self, include_empty: bool = True) -> list[int]:
        counts = Counter(map(int, self.assignments.tolist()))
        if include_empty:
            return [int(counts.get(k, 0)) for k in range(len(self.center_states) if self.center_states else self.K)]
        return [int(counts[k]) for k in sorted(counts)]

    def _extra_eval_fields(self, num_selected: int) -> dict:
        return {
            "num_selected_clients": int(num_selected),
            "cluster_sizes": self.cluster_sizes(),
            "assignment_interval": int(self.cfg.method.get("assignment_interval", 1)),
            "center_init": str(self.cfg.method.get("init", "random_perturb")),
        }

