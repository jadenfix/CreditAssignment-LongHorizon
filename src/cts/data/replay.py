"""Offline critique replay — the ONLY data interface trainers see.

Any upstream source (fixtures, teacher API outputs, dataset extractions) writes
into a directory of jsonl shards. Trainers open a :class:`ReplayShard` and
iterate :class:`Quadruple` objects. No training code ever talks to an API.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import Quadruple, dataset_hash


class ReplayShard:
    """A single jsonl file of :class:`Quadruple` records.

    Use :meth:`iter_quads` for streaming; :meth:`load_all` for in-memory use in
    tests / small experiments.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Replay shard missing: {self.path}")

    def iter_quads(self) -> Iterator[Quadruple]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield Quadruple.from_jsonl(line)

    def load_all(self) -> list[Quadruple]:
        return list(self.iter_quads())

    def load_split(self, split: str) -> list[Quadruple]:
        """Filter this shard's records by ``meta.split`` (for mixed-split files like fixtures)."""
        return [q for q in self.iter_quads() if q.meta.split == split]

    def __len__(self) -> int:
        n = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n


class ReplayDir:
    """A directory of jsonl shards. Splits are implicit: files named
    ``{split}.jsonl`` or ``{split}-*.jsonl``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Replay dir missing: {self.root}")

    def shards(self, split: str) -> list[ReplayShard]:
        matches = sorted(self.root.glob(f"{split}*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No shards for split={split!r} in {self.root}")
        return [ReplayShard(p) for p in matches]

    def load_split(self, split: str) -> list[Quadruple]:
        out: list[Quadruple] = []
        for shard in self.shards(split):
            out.extend(shard.load_all())
        return out

    def split_hash(self, split: str) -> str:
        return dataset_hash(self.load_split(split))


def write_jsonl(path: str | Path, quads: Iterable[Quadruple]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for q in quads:
            fh.write(q.to_jsonl())
            fh.write("\n")
            n += 1
    return n


def sample_minibatches(
    quads: list[Quadruple],
    batch_size: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[list[Quadruple]]:
    """Deterministic minibatch iterator keyed by ``seed``."""
    idx = list(range(len(quads)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            break
        yield [quads[i] for i in chunk]


def write_split_as_json(path: str | Path, quads: list[Quadruple]) -> None:
    """Debug helper: dump a pretty-printed JSON view of a split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([q.model_dump() for q in quads], indent=2), encoding="utf-8")
