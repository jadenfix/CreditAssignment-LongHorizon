from cts.data.replay import ReplayShard
from cts.data.schema import Quadruple, QuadrupleMeta, dataset_hash

FIX = "src/cts/data/fixtures/gsm8k_tiny.jsonl"


def test_roundtrip():
    q = Quadruple(x="p", y0="a", f="f", y1="b", meta=QuadrupleMeta(domain="math", task_id="t"))
    s = q.to_jsonl()
    r = Quadruple.from_jsonl(s)
    assert r == q


def test_fixture_loads_and_has_splits():
    shard = ReplayShard(FIX)
    all_q = shard.load_all()
    assert len(all_q) >= 8
    for split in ("train", "val", "test"):
        assert shard.load_split(split), f"no quads for split={split}"


def test_dataset_hash_stable():
    quads = ReplayShard(FIX).load_all()
    h1 = dataset_hash(quads)
    h2 = dataset_hash(list(reversed(quads)))
    assert h1 == h2, "hash must be order-independent"
