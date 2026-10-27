import numpy as np

from speech_scoring.scoring.confidence import conformal_halfwidth, reliability


def test_conformal_zero_when_perfect():
    y = np.array([1.0, 5.0, 9.0, 3.0])
    assert conformal_halfwidth(y, y.copy()) == 0.0


def test_conformal_positive_with_error():
    y = np.array([1.0, 5.0, 9.0, 3.0, 7.0])
    pred = y + np.array([0.5, -0.5, 1.0, -1.0, 0.2])
    assert conformal_halfwidth(y, pred) > 0


def test_reliability_high_on_clean_input():
    feats = {
        "accuracy__wer": 0.05,
        "accuracy__mean_word_prob": 0.92,
        "delivery__n_runs": 12,
        "delivery__mean_length_of_run": 6,
        "delivery__phonation_time_ratio": 0.65,
    }
    level, reasons = reliability(feats)
    assert level == "high" and reasons == []


def test_reliability_low_on_garbage_input():
    feats = {
        "accuracy__wer": 0.4,
        "accuracy__mean_word_prob": 0.3,
        "delivery__n_runs": 2,
        "delivery__mean_length_of_run": 2,
        "delivery__phonation_time_ratio": 0.1,
    }
    level, reasons = reliability(feats)
    assert level == "low" and len(reasons) >= 2
