"""Scoring metrics + bootstrap confidence intervals.

Pearson correlation is the standard metric on speechocean762, so it leads. QWK and
exact/adjacent agreement on the rounded 0-10 score are reported alongside.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def spearman(y_true, y_pred) -> float:
    if len(np.unique(y_pred)) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def pearson(y_true, y_pred) -> float:
    if len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def quadratic_weighted_kappa(y_true, y_pred, lo: int = 0, hi: int = 10) -> float:
    """QWK on integer-binned scores in [lo, hi]."""
    t = np.clip(np.round(np.asarray(y_true)), lo, hi).astype(int) - lo
    p = np.clip(np.round(np.asarray(y_pred)), lo, hi).astype(int) - lo
    n = hi - lo + 1
    obs = np.zeros((n, n))
    for a, b in zip(t, p):
        obs[a, b] += 1
    w = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            w[a, b] = (a - b) ** 2 / (n - 1) ** 2
    act = np.bincount(t, minlength=n)
    pre = np.bincount(p, minlength=n)
    exp = np.outer(act, pre) / max(1, obs.sum())
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else float("nan")


def exact_agreement(y_true, y_pred, lo: int = 0, hi: int = 10) -> float:
    """Fraction of samples whose rounded score exactly matches (rubric view)."""
    t = np.clip(np.round(np.asarray(y_true)), lo, hi)
    p = np.clip(np.round(np.asarray(y_pred)), lo, hi)
    return float(np.mean(t == p))


def adjacent_agreement(y_true, y_pred, lo: int = 0, hi: int = 10, tol: int = 1) -> float:
    """Fraction within `tol` score points (near-miss vs wild-miss diagnostic)."""
    t = np.clip(np.round(np.asarray(y_true)), lo, hi)
    p = np.clip(np.round(np.asarray(y_pred)), lo, hi)
    return float(np.mean(np.abs(t - p) <= tol))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "pearson": pearson(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "exact_agree": exact_agreement(y_true, y_pred),
        "adjacent_agree": adjacent_agreement(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def bootstrap_ci(
    y_true, y_pred, metric=pearson, n: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric over paired (true, pred)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        v = metric(y_true[idx], y_pred[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(vals, 100 * alpha / 2)),
        float(np.percentile(vals, 100 * (1 - alpha / 2))),
    )
