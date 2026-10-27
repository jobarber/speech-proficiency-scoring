import numpy as np

from speech_scoring.scoring import metrics


def test_spearman_perfect_rank():
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert metrics.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_qwk_identical_is_one():
    y = [1, 3, 5, 7, 9]
    assert abs(metrics.quadratic_weighted_kappa(y, y) - 1.0) < 1e-9


def test_exact_and_adjacent_agreement():
    y_true = [5, 5, 5, 5]
    y_pred = [5, 6, 7, 5]  # exact: 2/4 ; within 1: 3/4
    assert metrics.exact_agreement(y_true, y_pred) == 0.5
    assert metrics.adjacent_agreement(y_true, y_pred) == 0.75


def test_all_metrics_leads_with_pearson_and_has_agreement():
    m = metrics.all_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert list(m)[0] == "pearson"
    assert {"exact_agree", "adjacent_agree"} <= set(m)


def test_rmse_mae():
    assert metrics.rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert metrics.mae([1, 2, 3], [2, 3, 4]) == 1.0


def test_bootstrap_ci_bounds_order():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.5, size=40)
    lo, hi = metrics.bootstrap_ci(x, y, n=500)
    assert lo <= hi
