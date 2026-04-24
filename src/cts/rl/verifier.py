"""Task verifiers: math answer-match and code unit-test runner.

The code verifier runs candidate solutions in a subprocess with a CPU-time and
memory limit, so an infinite loop or rogue ``import os`` cannot hang the
pytest run. Untrusted code execution — keep the trust boundary obvious.
"""

from __future__ import annotations

import contextlib
import math
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Verifier(Protocol):
    def score(self, task_id: str, output: str) -> float: ...


# ---------- Math ----------

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(s: str) -> str | None:
    """Best-effort: take the last number in the string (GSM8K convention)."""
    matches = _NUMBER_RE.findall(s.replace(",", ""))
    return matches[-1] if matches else None


@dataclass
class MathVerifier:
    """Compares the last numeric token to a ground-truth map ``{task_id: answer}``."""

    gold: dict[str, str]
    tol: float = 1e-4

    def score(self, task_id: str, output: str) -> float:
        pred = extract_final_number(output)
        gold = self.gold.get(task_id)
        if pred is None or gold is None:
            return 0.0
        try:
            return float(math.isclose(float(pred), float(gold), rel_tol=0.0, abs_tol=self.tol))
        except ValueError:
            return 0.0


# ---------- Code ----------


@dataclass
class CodeVerifier:
    """Runs candidate Python code against a bank of unit tests per task.

    ``tests[task_id]`` is a list of assertion strings that will be appended
    after the candidate definition. The test passes if all assertions run
    without raising.
    """

    tests: dict[str, list[str]]
    timeout_sec: float = 3.0
    memory_mb: int = 256

    def _run(self, code: str) -> bool:
        def limit_resources() -> None:  # pragma: no cover — subprocess init
            cpu_sec = int(self.timeout_sec) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
            bytes_ = self.memory_mb * 1024 * 1024
            with contextlib.suppress(Exception):
                resource.setrlimit(resource.RLIMIT_AS, (bytes_, bytes_))

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            path = Path(fh.name)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                timeout=self.timeout_sec,
                preexec_fn=limit_resources if sys.platform != "win32" else None,
                text=True,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            with contextlib.suppress(Exception):
                path.unlink(missing_ok=True)

    def score(self, task_id: str, output: str) -> float:
        tests = self.tests.get(task_id)
        if not tests:
            return 0.0
        program = output.strip() + "\n\n" + "\n".join(tests) + "\n"
        return 1.0 if self._run(program) else 0.0
