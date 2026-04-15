"""Teacher-model generators. OPTIONAL upstream producers that write quadruple
shards into :mod:`cts.data.replay`. Nothing under :mod:`cts.train` calls these
directly — the replay shard is the only interface the trainer sees.
"""

from .base import TeacherProtocol, TeacherQuad

__all__ = ["TeacherProtocol", "TeacherQuad"]
