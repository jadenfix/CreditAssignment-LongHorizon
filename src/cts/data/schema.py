"""Core data schema: the Quadruple (x, y0, f, y1) is the only currency.

Every loader, trainer, and evaluator flows quadruples. Nothing else.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorTag(str, Enum):
    LOCAL_ARITHMETIC = "local_arithmetic"
    MISSING_CASE = "missing_case"
    INVALID_ASSUMPTION = "invalid_assumption"
    WRONG_DECOMPOSITION = "wrong_decomposition"
    TOOL_MISUSE = "tool_misuse"
    STATE_TRACKING = "state_tracking"
    STYLE_ONLY = "style_only"


class QuadrupleMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    domain: str = Field(description="'math' | 'code' | 'toy'")
    task_id: str
    split: str = Field(default="train", description="'train' | 'val' | 'test' | 'transfer'")
    horizon: int | None = None
    edit_distance: int | None = None
    error_tags: list[ErrorTag] = Field(default_factory=list)
    is_human_audited: bool = False
    teacher: str | None = None
    y0_correct: bool = False
    y1_correct: bool | None = None


class Quadruple(BaseModel):
    """A single (prompt, flawed attempt, critique, corrected revision) example."""

    model_config = ConfigDict(extra="forbid")

    x: str
    y0: str
    f: str
    y1: str
    meta: QuadrupleMeta

    def content_hash(self) -> str:
        payload = json.dumps(
            {
                "x": self.x,
                "y0": self.y0,
                "f": self.f,
                "y1": self.y1,
                "task_id": self.meta.task_id,
                "domain": self.meta.domain,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_jsonl(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> Quadruple:
        return cls.model_validate_json(line)


def dataset_hash(quads: list[Quadruple]) -> str:
    """Stable hash of a dataset shard, used for fairness guards and test-set freezing."""
    h = hashlib.sha256()
    for q in sorted(quads, key=lambda q: q.meta.task_id):
        h.update(q.content_hash().encode())
    return h.hexdigest()[:16]


def coerce_meta(raw: dict[str, Any]) -> QuadrupleMeta:
    return QuadrupleMeta.model_validate(raw)
