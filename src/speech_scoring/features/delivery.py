"""Delivery / fluency features from word timing + pause structure.

Rates, pause structure, run lengths and repetitions. All features are scalars in
human-readable units (words per second, seconds, counts), so they can be quoted
back to a learner directly.
"""

from __future__ import annotations

import re

import numpy as np
import pronouncing

from ..recording import Recording
from ..textutils import normalize
from .base import ns

_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def syllable_count(word: str) -> int:
    """CMUdict syllable count with a vowel-group fallback for OOV words."""
    w = normalize(word)
    if not w:
        return 0
    phones = pronouncing.phones_for_word(w)
    if phones:
        return max(1, pronouncing.syllable_count(phones[0]))
    return max(1, len(_VOWEL_GROUP.findall(w)))


def _sd(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    return float(xs.std(ddof=1)) if xs.size > 1 else 0.0


def extract(rec: Recording, total_duration: float) -> dict[str, float]:
    g = "delivery"
    words = rec.words
    n_words = len(words)
    if n_words == 0 or total_duration <= 0:
        # graceful zero-vector for silent/empty input
        return {ns(g, k): 0.0 for k in _FEATURE_KEYS}

    n_syll = sum(syllable_count(w.word) for w in words)
    phonation = rec.phonation_time
    pause_durs = [p.duration for p in rec.pauses]
    pause_total = float(sum(pause_durs))
    run_lens = [len(r) for r in rec.runs]
    interword = [words[i + 1].start - words[i].end for i in range(n_words - 1)]

    # repetitions: consecutive identical normalized tokens (disfluency marker)
    norm = [normalize(w.word) for w in words]
    repetitions = sum(1 for i in range(1, n_words) if norm[i] and norm[i] == norm[i - 1])

    feats = {
        "speech_rate_wps": n_words / total_duration,
        "speech_rate_sps": n_syll / total_duration,
        "articulation_rate_wps": n_words / phonation if phonation > 0 else 0.0,
        "articulation_rate_sps": n_syll / phonation if phonation > 0 else 0.0,
        "phonation_time_ratio": phonation / total_duration,
        "silence_ratio": pause_total / total_duration,
        "n_pauses": float(len(rec.pauses)),
        "pause_rate_per_min": len(rec.pauses) / total_duration * 60.0,
        "pause_total_s": pause_total,
        "pause_mean_s": float(np.mean(pause_durs)) if pause_durs else 0.0,
        "pause_sd_s": _sd(pause_durs),
        "n_long_pauses": float(sum(1 for p in rec.pauses if p.is_long)),
        "long_pause_ratio": (sum(1 for p in rec.pauses if p.is_long) / len(rec.pauses))
        if rec.pauses
        else 0.0,
        "mean_length_of_run": float(np.mean(run_lens)) if run_lens else 0.0,
        "run_len_sd": _sd(run_lens),
        "n_runs": float(len(rec.runs)),
        "interword_interval_sd": _sd(interword),
        "repetition_count": float(repetitions),
        "repetition_rate": repetitions / n_words,
        "leading_silence_s": rec.speech_start,
        "mean_word_duration_s": phonation / n_words,
    }
    return {ns(g, k): float(v) for k, v in feats.items()}


_FEATURE_KEYS = (
    "speech_rate_wps",
    "speech_rate_sps",
    "articulation_rate_wps",
    "articulation_rate_sps",
    "phonation_time_ratio",
    "silence_ratio",
    "n_pauses",
    "pause_rate_per_min",
    "pause_total_s",
    "pause_mean_s",
    "pause_sd_s",
    "n_long_pauses",
    "long_pause_ratio",
    "mean_length_of_run",
    "run_len_sd",
    "n_runs",
    "interword_interval_sd",
    "repetition_count",
    "repetition_rate",
    "leading_silence_s",
    "mean_word_duration_s",
)
