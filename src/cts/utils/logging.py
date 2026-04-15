"""Tiny rich/print-based logger; wandb left optional."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        record = {"step": step, "t": time.time(), **{k: float(v) for k, v in metrics.items()}}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
