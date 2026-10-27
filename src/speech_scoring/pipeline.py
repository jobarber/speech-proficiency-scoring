"""End to end: features -> multi-head score -> interval, reliability, grounded feedback.

- `score_corpus` scores cached features (CPU only).
- `score_audio` scores a new recording against its prompt (runs Whisper + alignment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .constants import SCORE_HI, SCORE_LO, SUBSCORES
from .feedback.generate import generate, maybe_rephrase
from .scoring.confidence import reliability
from .scoring.final_model import ScorerBundle, load_model


@dataclass
class ScoringResult:
    utt_id: str
    total: float
    interval: tuple[float, float]
    reliability: str
    reliability_reasons: list[str]
    subscores: dict[str, float]
    feedback: str
    top_contributions: dict[str, float] = field(default_factory=dict)


def score_row(
    bundle: ScorerBundle, X_row: pd.DataFrame, utt_id: str, settings: Settings | None = None
) -> ScoringResult:
    settings = settings or get_settings()
    X_row = X_row[bundle.feature_columns]
    model = bundle.model
    total = float(model.predict(X_row)[0])
    subs = model.predict_subscores(X_row).iloc[0].to_dict()
    contribs = model.contributions(X_row)
    fb = maybe_rephrase(generate(utt_id, total, subs, contribs, bundle.subscore_stats), settings)

    q = bundle.conformal_q
    interval = (
        float(np.clip(total - q, SCORE_LO, SCORE_HI)),
        float(np.clip(total + q, SCORE_LO, SCORE_HI)),
    )
    rel, reasons = reliability(X_row.iloc[0].to_dict())

    top = {}
    for sub in SUBSCORES:
        row = contribs[sub].iloc[0].sort_values(key=abs, ascending=False)
        top.update({feat: float(val) for feat, val in row.head(2).items()})
    return ScoringResult(utt_id, total, interval, rel, reasons, subs, fb.text, top)


def score_corpus(settings: Settings | None = None, split: str = "test") -> pd.DataFrame:
    from .features.extract import load_features

    settings = settings or get_settings()
    df = load_features(settings)
    df = df[df["split"] == split]
    bundle = load_model(settings)
    rows = []
    for uid in df.index:
        r = score_row(bundle, df.loc[[uid]], uid, settings)
        rows.append(
            {
                "utt_id": uid,
                "total": r.total,
                "ci_low": r.interval[0],
                "ci_high": r.interval[1],
                "reliability": r.reliability,
                **r.subscores,
                "feedback": r.feedback,
            }
        )
    out = pd.DataFrame(rows).set_index("utt_id")
    out.to_csv(settings.output_dir / f"scores_{split}.csv")
    return out


def score_audio(
    audio_path: str | Path, prompt: str, settings: Settings | None = None
) -> ScoringResult:
    """Score one recording of someone reading `prompt` aloud."""
    from .asr import WhisperTranscriber
    from .data import Utterance
    from .features import gop
    from .features.extract import features_for

    settings = settings or get_settings()
    words = WhisperTranscriber(settings.whisper_model).transcribe(audio_path)
    scores = gop.ForcedAligner().word_scores(audio_path, prompt)
    utt = Utterance("input", "input", "input", Path(audio_path), prompt, {})
    X_row = pd.DataFrame([features_for(utt, words, scores)])
    return score_row(load_model(settings), X_row, "input", settings)
