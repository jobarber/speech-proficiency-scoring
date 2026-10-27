"""Shared audio loading with peak normalization.

Whisper input is intentionally left raw (it normalizes internally); this loader
is for the feature/prosody path, where peak normalization improves cross-speaker
comparability of energy-based measures. Pitch/energy features are additionally
expressed in speaker-relative units (semitones vs median F0, dB vs max).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SR = 16000


def load_audio(path: str | Path, sr: int = SR, normalize: bool = True) -> tuple[np.ndarray, int]:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if normalize and y.size and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr
