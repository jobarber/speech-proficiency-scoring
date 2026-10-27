"""Goodness-of-pronunciation (GOP) features via forced alignment.

The prompt is force-aligned to the audio with torchaudio's MMS_FA CTC model, and
each expected word is scored by its mean frame posterior along the alignment. A
word that was misread or mispronounced aligns poorly and scores low. No Kaldi,
MFA, or espeak dependency.
"""

from __future__ import annotations

import numpy as np

from ..audioio import load_audio
from ..textutils import tokens
from .base import ns

LOW_GOP = 0.5  # per-word alignment score below which a word counts as poorly realized
KEYS = ("gop_mean", "gop_min", "gop_std", "gop_low_frac")


class ForcedAligner:
    """Lazy wrapper around torchaudio's MMS forced-alignment bundle."""

    def __init__(self):
        self._ready = False

    def _load(self):
        if self._ready:
            return
        import torch
        import torchaudio

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = torchaudio.pipelines.MMS_FA
        self.model = bundle.get_model().to(self.device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.sample_rate = bundle.sample_rate
        self._ready = True

    def word_scores(self, audio_path, reference_text: str) -> list[float]:
        """Per-word forced-alignment scores for the prompt's words."""
        self._load()
        words = [w for w in tokens(reference_text) if w.isascii() and any(c.isalpha() for c in w)]
        words = ["".join(c for c in w if c.isalpha()) for w in words]
        if not words:
            return []
        y, _ = load_audio(audio_path, sr=self.sample_rate, normalize=True)
        wav = self.torch.tensor(y, dtype=self.torch.float32).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            emission, _ = self.model(wav)
            token_spans = self.aligner(emission[0], self.tokenizer(words))
        scores = []
        for spans in token_spans:
            length = sum(s.end - s.start for s in spans)
            if length > 0:
                scores.append(float(sum(s.score * (s.end - s.start) for s in spans) / length))
        return scores


def features(scores: list[float]) -> dict[str, float]:
    g = "accuracy"
    if not scores:
        return {ns(g, k): 0.0 for k in KEYS}
    s = np.asarray(scores, dtype=float)
    return {
        ns(g, "gop_mean"): float(s.mean()),
        ns(g, "gop_min"): float(s.min()),
        ns(g, "gop_std"): float(s.std(ddof=1)) if s.size > 1 else 0.0,
        ns(g, "gop_low_frac"): float((s < LOW_GOP).mean()),
    }
