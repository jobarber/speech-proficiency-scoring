"""Whisper transcription with word-level timestamps and per-word confidence.

Word timings drive the fluency features; per-word probability is a cheap acoustic
signal for pronunciation. Heavy imports are local so the package imports without
the `gpu` extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float
    prob: float


class WhisperTranscriber:
    """Lazy-loaded Whisper model."""

    def __init__(self, model_name: str = "small.en", device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            import whisper

            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = whisper.load_model(self.model_name, device=self._device)
        return self._model

    def transcribe(self, audio_path: str | Path) -> list[Word]:
        """Greedy decoding without conditioning on previous text, for determinism and
        to limit the language model's tendency to tidy away disfluencies."""
        result = self._load().transcribe(
            str(audio_path),
            language="en",
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=False,
            fp16=(self._device == "cuda"),
        )
        return [
            Word(
                word=w["word"].strip(),
                start=float(w["start"]),
                end=float(w["end"]),
                prob=float(w.get("probability", 0.0)),
            )
            for seg in result.get("segments", [])
            for w in seg.get("words", [])
        ]
