"""Client sampling helpers."""
from __future__ import annotations

import numpy as np


def sample_client_ids(num_clients: int, frac: float, rng: np.random.Generator) -> list[int]:
    m = max(1, int(round(float(frac) * int(num_clients))))
    m = min(m, int(num_clients))
    return rng.choice(int(num_clients), size=m, replace=False).astype(int).tolist()
