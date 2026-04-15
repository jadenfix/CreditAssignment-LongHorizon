from cts.data.replay import ReplayShard
from cts.data.splits import assign_splits
from cts.data.taxonomy import is_local, is_structural, tag_histogram

FIX = "src/cts/data/fixtures/gsm8k_tiny.jsonl"


def test_assign_splits_deterministic():
    quads = ReplayShard(FIX).load_all()
    a = assign_splits(quads)
    b = assign_splits(quads)
    assert [q.meta.split for q in a] == [q.meta.split for q in b]


def test_taxonomy_classifies_fixtures():
    quads = ReplayShard(FIX).load_all()
    n_struct = sum(1 for q in quads if is_structural(q))
    n_local = sum(1 for q in quads if is_local(q))
    # fixtures are hand-labeled and diverse — both categories should be populated
    assert n_struct > 0
    assert n_local > 0
    hist = tag_histogram(quads)
    assert isinstance(hist, dict) and hist
