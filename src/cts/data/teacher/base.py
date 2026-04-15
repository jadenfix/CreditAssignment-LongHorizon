"""Common teacher protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TeacherQuad:
    y0: str
    f: str
    y1: str


@runtime_checkable
class TeacherProtocol(Protocol):
    name: str

    def generate(self, x: str, *, domain: str) -> TeacherQuad: ...
