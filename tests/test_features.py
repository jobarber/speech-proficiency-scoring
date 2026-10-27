import numpy as np

from speech_scoring.asr import Word
from speech_scoring.features import accuracy, delivery, gop, prosody
from speech_scoring.recording import build_recording


def _rec():
    words = [
        Word("the", 0.0, 0.5, 0.9),
        Word("cat", 0.5, 1.0, 0.9),
        Word("sat", 1.0, 1.5, 0.9),
        Word("down", 1.5, 2.0, 0.9),
    ]
    return build_recording("s", words, "The cat sat down.")


def test_delivery_speech_rate():
    rec = _rec()
    f = delivery.extract(rec, total_duration=2.0)
    # 4 words over 2 seconds -> 2.0 wps
    assert abs(f["delivery__speech_rate_wps"] - 2.0) < 1e-9
    assert f["delivery__phonation_time_ratio"] == 1.0  # no gaps


def test_delivery_empty_is_zero_no_nan():
    rec = build_recording("s", [], "ref")
    f = delivery.extract(rec, total_duration=5.0)
    assert all(v == 0.0 for v in f.values())
    assert not any(np.isnan(v) for v in f.values())


def test_accuracy_perfect_reading():
    rec = _rec()
    f = accuracy.extract(rec, "the cat sat down")
    assert f["accuracy__wer"] == 0.0
    assert f["accuracy__mean_word_prob"] > 0.8


def test_gop_empty_is_zero():
    f = gop.features([])
    assert set(f) == {f"accuracy__{k}" for k in gop.KEYS}
    assert all(v == 0.0 for v in f.values())


def test_gop_aggregates_known_scores():
    f = gop.features([0.9, 0.8, 0.3, 0.95])
    assert abs(f["accuracy__gop_mean"] - 0.7375) < 1e-6
    assert f["accuracy__gop_min"] == 0.3
    assert f["accuracy__gop_low_frac"] == 0.25


def test_prosody_on_synthetic_tone():
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)  # gliding pitch
    y = 0.5 * np.sin(2 * np.pi * np.cumsum(f0) / sr).astype(np.float32)
    f = prosody.extract(_rec(), y, sr)
    assert f["prosody__voiced_frac"] > 0.8
    assert f["prosody__pitch_range_semitone"] > 1.0
