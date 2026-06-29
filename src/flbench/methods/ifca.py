"""IFCA baseline.

Iterative Federated Clustering Algorithm:
  1. Keep K cluster models.
  2. Each selected client picks the model with the lowest local assignment loss.
  3. The client trains from that selected model.
  4. Server aggregates updates within each predicted cluster.

This implementation follows the same BaseRunner / FLClient / evaluator interfaces
used by FedAvg and FeSEM in this benchmark.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from flbench.core.evaluation import evaluate_cluster_models, evaluate_global_model
from flbench.core.sampling import sample_client_ids
from flbench.methods.base import BaseRunner
from flbench.utils.state_dict import (
    StateDict,
    apply_delta,
    clone_state,
    flatten_delta,
    perturb_state,
    weighted_average_deltas,
    weighted_average_states,
)


class IFCARunner(BaseRunner):
    name = "ifca"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.K = int(self.cfg.method.get("num_clusters", 1))
        if self.K < 1:
            raise ValueError("IFCA requires method.num_clusters >= 1")

        self.center_states: list[StateDict] = []
        self.assignments = np.zeros(self.num_clients, dtype=np.int64)
        self.global_state: StateDict | None = None

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

        self.global_state = self.init_model_state()

        # Optional FedAvg warm-up. These rounds are included in total_rounds.
        for r in range(1, warmup_rounds + 1):
            selected = sample_client_ids(
                self.num_clients,
                float(self.cfg.train.client_frac),
                self.rng,
            )
            self.global_state = self._fedavg_step(self.global_state, selected)

            if r % eval_every == 0 or r == warmup_rounds:
                metrics = evaluate_global_model(
                    clients=self.clients,
                    model_state=self.global_state,
                    model_fn=self.model_fn,
                    num_classes=self.num_classes,
                    split=str(self.cfg.eval.get("test_split", "test")),
                )
                self.log_metrics(
                    r,
                    "warmup",
                    metrics,
                    extra={
                        "num_selected_clients": len(selected),
                        "cluster_sizes": [self.num_clients],
                        "num_empty_clusters": 0,
                    },
                )

        self._initialize_centers(self.global_state)
        self.assign_all_clients()

        if warmup_rounds == total_rounds:
            metrics = evaluate_cluster_models(
                self.clients,
                self.center_states,
                self.assignments,
                self.model_fn,
                self.num_classes,
                split=str(self.cfg.eval.get("test_split", "test")),
            )
            self.log_metrics(
                total_rounds,
                "final",
                metrics,
                extra=self._extra_eval_fields(num_selected=0, updated_clusters=[]),
            )
            return self.save_and_summarize()

        for r in range(warmup_rounds + 1, total_rounds + 1):
            selected = sample_client_ids(
                self.num_clients,
                float(self.cfg.train.client_frac),
                self.rng,
            )

            updated_clusters = self._ifca_step(selected)

            if r % eval_every == 0 or r == total_rounds:
                if bool(self.cfg.method.get("eval_assign_all", True)):
                    self.assign_all_clients()

                metrics = evaluate_cluster_models(
                    self.clients,
                    self.center_states,
                    self.assignments,
                    self.model_fn,
                    self.num_classes,
                    split=str(self.cfg.eval.get("test_split", "test")),
                )
                self.log_metrics(
                    r,
                    "clustered",
                    metrics,
                    extra=self._extra_eval_fields(
                        num_selected=len(selected),
                        updated_clusters=updated_clusters,
                    ),
                )

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

    def _initialize_centers(self, base_state: StateDict) -> None:
        init = str(self.cfg.method.get("init", "random_perturb")).lower()

        if init == "same":
            self.center_states = [clone_state(base_state) for _ in range(self.K)]

        elif init == "random_perturb":
            sigma = float(self.cfg.method.get("center_perturb_sigma", 0.01))
            self.center_states = [
                perturb_state(base_state, sigma=sigma, seed=self.seed + 1000 + k)
                for k in range(self.K)
            ]

        elif init == "warmup_kmeans":
            self.center_states = self._warmup_kmeans_init(base_state)

        else:
            raise ValueError(f"unknown IFCA init: {init}")

        print(f"[ifca] initialized {len(self.center_states)} centers using init={init}")

    def _warmup_kmeans_init(self, base_state: StateDict) -> list[StateDict]:
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
                proximal_mu=float(self.cfg.train.get("proximal_mu", 0.0)),
            )
            vectors.append(flatten_delta(result.delta).numpy())
            new_states.append(result.new_state)
            weights.append(result.num_samples)

        self.model_downloads += len(client_ids)
        self.model_uploads += len(client_ids)

        if len(client_ids) < self.K:
            sigma = float(self.cfg.method.get("center_perturb_sigma", 0.01))
            return [
                perturb_state(base_state, sigma=sigma, seed=self.seed + 2000 + k)
                for k in range(self.K)
            ]

        x = np.stack(vectors, axis=0)
        km = KMeans(n_clusters=self.K, n_init=10, random_state=self.seed + 2000)
        labels = km.fit_predict(x)

        centers: list[StateDict] = []

        for k in range(self.K):
            member_pos = [pos for pos, label in enumerate(labels) if int(label) == k]

            if member_pos:
                states_k = [new_states[pos] for pos in member_pos]
                weights_k = [weights[pos] for pos in member_pos]
                centers.append(weighted_average_states(states_k, weights_k))
            else:
                centers.append(
                    perturb_state(
                        base_state,
                        sigma=float(self.cfg.method.get("center_perturb_sigma", 0.01)),
                        seed=self.seed + 3000 + k,
                    )
                )

        return centers

    def _ifca_step(self, selected: list[int]) -> list[int]:
        deltas_by_cluster: dict[int, list[StateDict]] = {k: [] for k in range(self.K)}
        weights_by_cluster: dict[int, list[int]] = {k: [] for k in range(self.K)}

        for idx in selected:
            k = self.best_center_for_client(idx)
            self.assignments[idx] = k

            result = self.clients[idx].train(
                model_state=self.center_states[k],
                model_fn=self.model_fn,
                epochs=int(self.cfg.train.local_epochs),
                lr=float(self.cfg.train.lr),
                optimizer_name=str(self.cfg.train.get("optimizer", "sgd")),
                momentum=float(self.cfg.train.get("momentum", 0.9)),
                weight_decay=float(self.cfg.train.get("weight_decay", 0.0)),
                proximal_mu=float(self.cfg.train.get("proximal_mu", 0.0)),
            )

            deltas_by_cluster[k].append(result.delta)
            weights_by_cluster[k].append(result.num_samples)

            self.model_downloads += 1
            self.model_uploads += 1

        updated_clusters: list[int] = []

        for k in range(self.K):
            if not deltas_by_cluster[k]:
                continue

            avg_delta = weighted_average_deltas(
                deltas_by_cluster[k],
                weights_by_cluster[k],
            )
            self.center_states[k] = apply_delta(self.center_states[k], avg_delta)
            updated_clusters.append(k)

        return updated_clusters

    def best_center_for_client(self, client_idx: int) -> int:
        split = str(
            self.cfg.method.get(
                "assignment_split",
                self.cfg.eval.get("assignment_split", "val"),
            )
        )

        losses = np.asarray(
            [
                self.clients[client_idx].loss_on_model(
                    state,
                    self.model_fn,
                    split=split,
                )
                for state in self.center_states
            ],
            dtype=np.float64,
        )

        self.assignment_model_evals += self.K

        finite_mask = np.isfinite(losses)
        if not np.any(finite_mask):
            current = int(self.assignments[client_idx])
            if 0 <= current < self.K:
                return current
            return 0

        safe_losses = np.where(finite_mask, losses, np.inf)
        min_loss = float(safe_losses.min())
        candidates = np.flatnonzero(np.isclose(safe_losses, min_loss, rtol=1e-10, atol=1e-12))

        # Deterministic tie handling: keep the current cluster if it is tied.
        current = int(self.assignments[client_idx])
        if current in candidates:
            return current

        if candidates.size:
            return int(candidates[0])

        return int(np.argmin(safe_losses))

    def assign_all_clients(self) -> None:
        new_assignments = np.zeros(self.num_clients, dtype=np.int64)

        for idx in range(self.num_clients):
            new_assignments[idx] = self.best_center_for_client(idx)

        self.assignments = new_assignments
        print(f"[ifca] assignment cluster sizes: {self.cluster_sizes()}")

    def cluster_sizes(self, include_empty: bool = True) -> list[int]:
        counts = Counter(map(int, self.assignments.tolist()))

        if include_empty:
            return [int(counts.get(k, 0)) for k in range(self.K)]

        return [int(counts[k]) for k in sorted(counts)]

    def _extra_eval_fields(self, num_selected: int, updated_clusters: list[int]) -> dict:
        sizes = self.cluster_sizes(include_empty=True)

        return {
            "num_selected_clients": int(num_selected),
            "cluster_sizes": sizes,
            "num_empty_clusters": int(sum(1 for s in sizes if s == 0)),
            "updated_clusters": updated_clusters,
            "assignment_split": str(
                self.cfg.method.get(
                    "assignment_split",
                    self.cfg.eval.get("assignment_split", "val"),
                )
            ),
            "center_init": str(self.cfg.method.get("init", "random_perturb")),
            "eval_assign_all": bool(self.cfg.method.get("eval_assign_all", True)),
        }
