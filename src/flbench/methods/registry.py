"""Method registry."""

from __future__ import annotations

from flbench.methods.fedavg import FedAvgRunner
from flbench.methods.fesem import FeSEMRunner
from flbench.methods.ifca import IFCARunner


RUNNERS = {
    FedAvgRunner.name: FedAvgRunner,
    FeSEMRunner.name: FeSEMRunner,
    IFCARunner.name: IFCARunner,
}


# Keep CFL optional so a broken/experimental cfl.py does not block IFCA/FedAvg/FeSEM.
try:
    from flbench.methods.cfl import CFLRunner

    RUNNERS[CFLRunner.name] = CFLRunner
except Exception as exc:  # pragma: no cover
    _CFL_IMPORT_ERROR = exc
else:
    _CFL_IMPORT_ERROR = None


def build_runner(method_name: str, *args, **kwargs):
    name = str(method_name).lower()

    if name not in RUNNERS:
        available = sorted(RUNNERS)
        if name == "cfl" and _CFL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "CFLRunner failed to import. Fix src/flbench/methods/cfl.py first. "
                f"Original error: {_CFL_IMPORT_ERROR}"
            )

        raise ValueError(f"unsupported method: {method_name}. Available: {available}")

    return RUNNERS[name](*args, **kwargs)

