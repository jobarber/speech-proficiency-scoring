"""Evidence-grounded learner feedback.

The feedback is generated from the scoring model's own outputs -- the predicted
analytic subscores and the per-feature contributions of the multi-head model --
so the explanation comes from the scoring process itself rather than from a
separate post-hoc explainer. An optional LLM can rephrase this grounded evidence
into smoother prose when an API key is configured.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..constants import SUBSCORES

# Human-readable phrasing per feature: (label, higher_is_better).
FEATURE_PHRASES: dict[str, tuple[str, bool]] = {
    "delivery__speech_rate_wps": ("speaking rate", True),
    "delivery__articulation_rate_wps": ("articulation rate", True),
    "delivery__phonation_time_ratio": ("proportion of time spent speaking", True),
    "delivery__silence_ratio": ("amount of silence", False),
    "delivery__mean_length_of_run": ("length of fluent runs between pauses", True),
    "delivery__n_long_pauses": ("number of long pauses", False),
    "delivery__long_pause_ratio": ("share of long pauses", False),
    "delivery__pause_rate_per_min": ("pause frequency", False),
    "delivery__repetition_rate": ("rate of repetitions and restarts", False),
    "accuracy__wer": ("word-error rate against the prompt", False),
    "accuracy__gop_mean": ("average word pronunciation score", True),
    "accuracy__gop_low_frac": ("share of poorly pronounced words", False),
    "accuracy__gop_min": ("minimum word pronunciation score", True),
    "accuracy__mean_word_prob": ("average word clarity", True),
    "accuracy__low_conf_frac": ("share of unclear words", False),
    "accuracy__min_word_prob": ("minimum word clarity", True),
    "prosody__pitch_range_semitone": ("intonation range", True),
    "prosody__pitch_sd_semitone": ("pitch variation", True),
    "prosody__energy_dynamic_range_db": ("loudness dynamics", True),
    "prosody__boundary_pause_ratio": ("pausing at sentence boundaries", True),
    "prosody__internal_long_pause_count": ("number of long pauses mid-sentence", False),
    "prosody__npvi_word_duration": ("rhythm variability", True),
}

SUBSCORE_LABELS = {
    "accuracy": "Accuracy (pronunciation & reading)",
    "fluency": "Fluency (pacing & pausing)",
    "prosodic": "Prosody (intonation & rhythm)",
}


def _band(score: float) -> str:
    """Band names follow the corpus rubric for sentence-level accuracy."""
    if score >= 8.5:
        return "excellent"
    if score >= 6.5:
        return "good"
    if score >= 4.5:
        return "understandable"
    if score >= 2.5:
        return "poor"
    return "very poor"


@dataclass
class Feedback:
    sample_id: str
    overall: float
    subscores: dict[str, float]
    strengths: list[str]
    growth: list[str]
    text: str


def _top_features(contrib_row: pd.Series, positive: bool, k: int = 2) -> list[str]:
    """Top human-named drivers in the requested direction (skip unnamed features)."""
    ordered = contrib_row.sort_values(ascending=not positive)
    phrases = []
    for feat, val in ordered.items():
        if (positive and val <= 0) or (not positive and val >= 0):
            break
        if feat in FEATURE_PHRASES:  # only cite explainable features
            label, higher_is_better = FEATURE_PHRASES[feat]
            if positive:  # a strength: the value is high if higher is better, else low
                phrases.append(f"a {'good' if higher_is_better else 'low'} {label}")
            else:  # a weakness: the value is low if higher is better, else high
                phrases.append(f"a {'low' if higher_is_better else 'high'} {label}")
        if len(phrases) >= k:
            break
    return phrases


def generate(
    sample_id: str,
    overall: float,
    subscores: dict[str, float],
    contributions: dict[str, pd.DataFrame],
    subscore_stats: dict[str, tuple[float, float]] | None = None,
) -> Feedback:
    """Build grounded feedback from predicted subscores + per-head contributions.

    `contributions[subscore]` is a 1-row DataFrame of per-feature contributions
    for this sample. `subscore_stats` gives (mean, std) per subscore so strength
    is judged relative to each dimension's own scale (they have different means).
    """

    def rel(sub):
        if subscore_stats and sub in subscore_stats:
            mu, sd = subscore_stats[sub]
            return (subscores[sub] - mu) / (sd or 1.0)
        return subscores[sub]

    ordered = sorted(subscores, key=rel, reverse=True)
    best_sub, worst_sub = ordered[0], ordered[-1]

    strengths, growth = [], []
    for sub in SUBSCORES:
        row = contributions[sub].iloc[0]
        if sub == best_sub:
            strengths = _top_features(row, positive=True, k=2)
        if sub == worst_sub:
            growth = _top_features(row, positive=False, k=2)

    band = _band(overall)
    parts = [f"Total score: **{overall:.1f}/10** ({band})."]
    parts.append(
        f"Your strongest area is **{SUBSCORE_LABELS[best_sub].split('(')[0].strip()}**"
        + (f", helped by {' and '.join(strengths)}." if strengths else ".")
    )
    if worst_sub != best_sub:
        parts.append(
            f"The biggest opportunity is **{SUBSCORE_LABELS[worst_sub].split('(')[0].strip()}**"
            + (f", held back by {' and '.join(growth)}." if growth else ".")
        )
    text = " ".join(parts)
    return Feedback(sample_id, overall, subscores, strengths, growth, text)


def maybe_rephrase(feedback: Feedback, settings) -> Feedback:
    """Optionally rephrase grounded evidence via an LLM (if API key set)."""
    if not getattr(settings, "anthropic_api_key", None):
        return feedback
    try:
        from .llm import rephrase  # noqa: PLC0415

        new_text = rephrase(feedback, settings)
        return Feedback(
            feedback.sample_id,
            feedback.overall,
            feedback.subscores,
            feedback.strengths,
            feedback.growth,
            new_text,
        )
    except Exception:
        return feedback  # never let optional LLM break the pipeline
