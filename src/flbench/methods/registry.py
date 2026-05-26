"""Method registry."""

from __future__ import annotations

from src.flbench.methods.fedavg import FedAvgRunner
from src.flbench.methods.fesem import FeSEMRunner
from src.flbench.methods.cfl import CFLRunner

RUNNERS = {
    FedAvgRunner.name: FedAvgRunner,
    FeSEMRunner.name: FeSEMRunner,
    CFLRunner.name: CFLRunner,
}


def build_runner(method_name: str, *args, **kwargs):
    name = str(method_name).lower()
    if name not in RUNNERS:
        raise ValueError(f"unsupported method: {method_name}. Available: {sorted(RUNNERS)}")
    return RUNNERS[name](*args, **kwargs)
