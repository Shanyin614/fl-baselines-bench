"""Aggregation primitives."""
from __future__ import annotations

from flbench.utils.state_dict import StateDict, apply_delta, weighted_average_deltas, weighted_average_states

__all__ = [
    "StateDict",
    "apply_delta",
    "weighted_average_deltas",
    "weighted_average_states",
]
