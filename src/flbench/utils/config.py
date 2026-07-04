"""Minimal YAML config helpers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        data = data or {}
        for key, value in data.items():
            self[key] = self._convert(value)

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._convert(value)

    def to_dict(self) -> dict[str, Any]:
        def rec(obj: Any) -> Any:
            if isinstance(obj, ConfigNode):
                return {k: rec(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [rec(v) for v in obj]
            return obj
        return rec(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of base updated by override."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

