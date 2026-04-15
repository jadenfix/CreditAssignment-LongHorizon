"""Anthropic Claude teacher. Requires ``ANTHROPIC_API_KEY`` in the env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .base import TeacherQuad

_INSTRUCTION = (
    "You will be given a problem. First produce a flawed initial attempt y0, then a short "
    "critique f naming the flaw, then a corrected revision y1. Separate them with the lines "
    "'###Y0', '###F', '###Y1'."
)


@dataclass
class AnthropicTeacher:
    name: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"

    def __post_init__(self) -> None:
        try:
            import anthropic
        except Exception as e:  # pragma: no cover
            raise ImportError("Install `cts[teacher]` for Anthropic teacher.") from e
        self._client = anthropic.Anthropic(api_key=os.environ[self.api_key_env])

    def generate(self, x: str, *, domain: str) -> TeacherQuad:
        prompt = f"{_INSTRUCTION}\n\nDomain: {domain}\nProblem: {x}"
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        return _parse(text)


def _parse(text: str) -> TeacherQuad:
    y0 = _between(text, "###Y0", "###F")
    f = _between(text, "###F", "###Y1")
    y1 = _after(text, "###Y1")
    return TeacherQuad(y0=y0.strip(), f=f.strip(), y1=y1.strip())


def _between(s: str, a: str, b: str) -> str:
    i = s.find(a)
    j = s.find(b)
    if i == -1 or j == -1 or j < i:
        return ""
    return s[i + len(a) : j]


def _after(s: str, a: str) -> str:
    i = s.find(a)
    return s[i + len(a) :] if i != -1 else ""
