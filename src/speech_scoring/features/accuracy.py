"""Accuracy / pronunciation features.

Three channels, from cheapest to most direct:
  - text deviation: edit rates between the prompt and the Whisper transcript.
  - recognizer confidence: Whisper's per-word probability. Whisper's language model
    tends to "repair" mispronounced words in the transcript, but its confidence
    still drops on them.
  - goodness of pronunciation (GOP): the prompt is force-aligned to the audio with a
    CTC acoustic model and each expected word gets an alignment score (see gop.py).
"""

from __future__ import annotations

import numpy as np

from ..recording import Recording
from ..textutils import error_rates
from .base import ns

LOW_CONF = 0.5
MED_CONF = 0.7


def extract(rec: Recording, reference_text: str) -> dict[str, float]:
    g = "accuracy"
    whisper_text = " ".join(w.word for w in rec.words)
    er = error_rates(reference_text, whisper_text)

    probs = np.array([w.prob for w in rec.words], dtype=float)
    if probs.size == 0:
        conf = {
            "mean_word_prob": 0.0,
            "min_word_prob": 0.0,
            "std_word_prob": 0.0,
            "low_conf_frac": 1.0,
            "med_conf_frac": 1.0,
        }
    else:
        conf = {
            "mean_word_prob": float(probs.mean()),
            "min_word_prob": float(probs.min()),
            "std_word_prob": float(probs.std(ddof=1)) if probs.size > 1 else 0.0,
            "low_conf_frac": float((probs < LOW_CONF).mean()),
            "med_conf_frac": float((probs < MED_CONF).mean()),
        }

    feats = {
        "wer": er["wer"],
        "cer": er["cer"],
        "sub_rate": er["sub_rate"],
        "del_rate": er["del_rate"],
        "ins_rate": er["ins_rate"],
        **conf,
    }
    return {ns(g, k): float(v) for k, v in feats.items()}
