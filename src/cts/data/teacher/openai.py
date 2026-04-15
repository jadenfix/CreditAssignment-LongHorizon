"""OpenAI teacher. Requires ``OPENAI_API_KEY`` in env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .anthropic import _parse  # reuse the same ###Y0/###F/###Y1 protocol
from .base import TeacherQuad

_INSTRUCTION = (
    "Given a problem, produce a flawed initial attempt y0, a short critique f, and a "
    "corrected revision y1. Use the separators '###Y0', '###F', '###Y1'."
)


@dataclass
class OpenAITeacher:
    name: str = "openai"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("Install `cts[teacher]` for OpenAI teacher.") from e
        self._client = OpenAI(api_key=os.environ[self.api_key_env])

    def generate(self, x: str, *, domain: str) -> TeacherQuad:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _INSTRUCTION},
                {"role": "user", "content": f"Domain: {domain}\nProblem: {x}"},
            ],
            temperature=0.7,
        )
        return _parse(resp.choices[0].message.content or "")
