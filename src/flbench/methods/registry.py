"""Method registry."""
from __future__ import annotations

from flbench.methods.fedavg import FedAvgRunner
from flbench.methods.fesem import FeSEMRunner
from flbench.methods.cfl import CFLRunner

RUNNERS = {
    FedAvgRunner.name: FedAvgRunner,
    FeSEMRunner.name: FeSEMRunner,
    "cfl": CFLRunner,

}


def build_runner(method_name: str, *args, **kwargs):
    name = str(method_name).lower()
    if name not in RUNNERS:
        raise ValueError(f"unsupported method: {method_name}. Available: {sorted(RUNNERS)}")
    return RUNNERS[name](*args, **kwargs)
