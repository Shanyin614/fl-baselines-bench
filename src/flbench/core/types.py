"""Shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClientMeta:
    client_id: int
    group_id: int | None
    train_indices: list[int] = field(repr=False)
    val_indices: list[int] = field(repr=False)
    test_indices: list[int] = field(repr=False)


@dataclass
class DatasetInfo:
    name: str
    num_classes: int
    split_file: str | None
    true_groups: list[list[int]]
    extra: dict[str, Any] = field(default_factory=dict)
