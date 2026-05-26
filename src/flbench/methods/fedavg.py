"""FedAvg baseline."""
from __future__ import annotations

from src.flbench.core.evaluation import evaluate_global_model
from src.flbench.core.sampling import sample_client_ids
from src.flbench.methods.base import BaseRunner
from src.flbench.utils.state_dict import apply_delta, weighted_average_deltas


class FedAvgRunner(BaseRunner):
    name = "fedavg"

    def run(self) -> dict:
        state = self.init_model_state()
        total_rounds = int(self.cfg.train.total_rounds)
        eval_every = int(self.cfg.eval.get("eval_every", 1))

        for r in range(1, total_rounds + 1):
            selected = sample_client_ids(self.num_clients, float(self.cfg.train.client_frac), self.rng)
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
            avg_delta = weighted_average_deltas(deltas, weights)
            state = apply_delta(state, avg_delta)

            if r % eval_every == 0 or r == total_rounds:
                metrics = evaluate_global_model(
                    clients=self.clients,
                    model_state=state,
                    model_fn=self.model_fn,
                    num_classes=self.num_classes,
                    split=str(self.cfg.eval.get("test_split", "test")),
                )
                self.log_metrics(r, "train", metrics, extra={"num_selected_clients": len(selected)})
        return self.save_and_summarize()
