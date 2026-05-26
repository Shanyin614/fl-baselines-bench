"""Split file IO."""
from __future__ import annotations

import json
from pathlib import Path

from src.flbench.core.types import ClientMeta


def save_client_metas(metas: list[ClientMeta], path: str | Path, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "clients": {
            str(meta.client_id): {
                "group_id": meta.group_id,
                "train_indices": list(map(int, meta.train_indices)),
                "val_indices": list(map(int, meta.val_indices)),
                "test_indices": list(map(int, meta.test_indices)),
            }
            for meta in metas
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def load_client_metas(path: str | Path) -> list[ClientMeta]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    clients = payload.get("clients", payload)
    metas: list[ClientMeta] = []
    for cid_str, item in sorted(clients.items(), key=lambda kv: int(kv[0])):
        metas.append(
            ClientMeta(
                client_id=int(cid_str),
                group_id=item.get("group_id", item.get("true_group", None)),
                train_indices=list(map(int, item["train_indices"])),
                val_indices=list(map(int, item["val_indices"])),
                test_indices=list(map(int, item["test_indices"])),
            )
        )
    return metas
