"""Byte-level tokenizer for nano-LM smoke paths.

Vocab of 128 = ASCII. Non-ASCII bytes are mapped to a single unknown id.
Reserved: 0 = PAD, 1 = BOS, 2 = EOS, 3 = SEP. First printable ASCII starts at 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

PAD, BOS, EOS, SEP = 0, 1, 2, 3
UNK = 127
_OFFSET = 4  # ASCII 32 (space) maps to 4+32 = 36; we map 0..UNK-_OFFSET-1 printable range


def _encode_char(c: str) -> int:
    b = ord(c)
    if 0 <= b < 124:
        return _OFFSET + b  # covers 0..123 → tokens 4..127; anything above -> UNK (127)
    return UNK


def _decode_token(t: int) -> str:
    if t == PAD:
        return ""
    if t == BOS:
        return "<s>"
    if t == EOS:
        return "</s>"
    if t == SEP:
        return "|"
    if t == UNK:
        return "?"
    return chr(max(0, min(123, t - _OFFSET)))


@dataclass
class ByteTokenizer:
    vocab_size: int = 128

    def encode(self, s: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [_encode_char(c) for c in s]
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: list[int] | jnp.ndarray) -> str:
        if not isinstance(ids, list):
            ids = [int(x) for x in ids]  # type: ignore[assignment]
        return "".join(_decode_token(t) for t in ids)

    def pad_to(self, ids: list[int], length: int) -> list[int]:
        if len(ids) >= length:
            return ids[:length]
        return ids + [PAD] * (length - len(ids))
