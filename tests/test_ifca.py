import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flbench.methods.ifca import IFCARunner


class DummyClient:
    def __init__(self, loss):
        self._loss = loss

    def loss_on_model(self, *args, **kwargs):
        return self._loss


def test_ifca_falls_back_when_all_losses_are_non_finite():
    runner = IFCARunner.__new__(IFCARunner)
    runner.cfg = type("Cfg", (), {"method": {}, "eval": {}})()
    runner.clients = [DummyClient(float("nan")) for _ in range(3)]
    runner.center_states = [object(), object(), object()]
    runner.assignments = np.array([1, 1, 1], dtype=np.int64)
    runner.K = 3
    runner.assignment_model_evals = 0

    assignment = runner.best_center_for_client(0)

    assert assignment == 1
