"""Per-sample confidence: a calibrated score interval + a reliability flag.

Two pieces, neither of them a separately trained model:
  - a conformal half-width from speaker-grouped out-of-fold residuals: a
    distribution-free interval with a coverage guarantee.
  - a rule-based reliability flag from input-quality signals (reading errors,
    recognizer confidence, amount of speech) to catch unusable input.
"""

from __future__ import annotations

import numpy as np

# Input-quality thresholds for the reliability flag.
WER_HIGH = 0.50
CONF_LOW = 0.60
MIN_WORDS = 3
MIN_PHONATION_RATIO = 0.30


def conformal_halfwidth(y_true: np.ndarray, oof_pred: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal half-width from out-of-fold absolute residuals.

    Gives a (1-alpha) interval: with the finite-sample correction the quantile
    level is ceil((n+1)(1-alpha))/n.
    """
    mask = ~np.isnan(oof_pred)
    res = np.abs(np.asarray(y_true)[mask] - np.asarray(oof_pred)[mask])
    n = len(res)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(res, level))


def reliability(features: dict[str, float]) -> tuple[str, list[str]]:
    """Rule-based reliability label (high/medium/low) + human reasons.

    `features` is a flat dict of namespaced feature -> value for one sample.
    """
    reasons = []
    if features.get("accuracy__wer", 0.0) > WER_HIGH:
        reasons.append("high reading-error rate")
    if features.get("accuracy__mean_word_prob", 1.0) < CONF_LOW:
        reasons.append("low ASR word confidence (unclear audio)")
    n_words = features.get("delivery__n_runs", 0.0) * features.get(
        "delivery__mean_length_of_run", 0.0
    )
    if n_words and n_words < MIN_WORDS:
        reasons.append("very little speech")
    if features.get("delivery__phonation_time_ratio", 1.0) < MIN_PHONATION_RATIO:
        reasons.append("mostly silence")
    level = "high" if not reasons else ("medium" if len(reasons) == 1 else "low")
    return level, reasons
