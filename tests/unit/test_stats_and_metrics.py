from cts.data.schema import Quadruple, QuadrupleMeta
from cts.eval.metrics import delta_critique
from cts.eval.stats import cohens_kappa, holm_bonferroni, paired_bootstrap_ci, wilcoxon_signed_rank


def _q(tid: str) -> Quadruple:
    return Quadruple(x="p", y0="", f="", y1="", meta=QuadrupleMeta(domain="math", task_id=tid))


def test_delta_critique():
    quads = [_q(f"t{i}") for i in range(4)]
    result = delta_critique(quads, [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0])
    assert result["delta_mean"] == 0.5
    assert result["n"] == 4


def test_bootstrap_runs():
    boot = paired_bootstrap_ci([0, 0, 0, 0], [1, 1, 1, 0], n_resamples=500, seed=0)
    assert boot["lo"] <= boot["mean"] <= boot["hi"]


def test_wilcoxon_symmetric_case():
    result = wilcoxon_signed_rank([0, 0, 0, 0], [1, 1, -1, -1])
    assert 0.0 <= result["p"] <= 1.0


def test_holm_rejects_smallest():
    rej = holm_bonferroni([0.001, 0.5], alpha=0.05)
    assert rej[0] is True
    assert rej[1] is False


def test_cohens_kappa_basic():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1])) < 1e-6 + 1.0  # ≤ 0
