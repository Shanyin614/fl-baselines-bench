"""CSV logging."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, output_dir: str | Path, output_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / output_name
        self.rows: list[dict[str, Any]] = []

    def log(self, row: dict[str, Any]) -> None:
        serializable = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple, dict)):
                serializable[key] = json.dumps(value, ensure_ascii=False)
            else:
                serializable[key] = value
        self.rows.append(serializable)

    def save(self) -> Path:
        if not self.rows:
            self.path.write_text("", encoding="utf-8")
            return self.path
        fieldnames: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        return self.path
