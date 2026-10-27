"""Prosody features (suprasegmental).

Pitch (F0) variation, intensity dynamics, rhythm variability, and whether pauses land
at sentence boundaries or mid-sentence. Boundary mapping assumes Whisper's word order
tracks the prompt, which holds for read speech with low WER."""

from __future__ import annotations

import librosa
import numpy as np

from ..recording import Recording
from ..textutils import tokens
from .base import ns

FMIN, FMAX = 65.0, 400.0


def _npvi(durations) -> float:
    """Normalized Pairwise Variability Index -- a standard rhythm metric."""
    d = np.asarray(durations, dtype=float)
    d = d[d > 0]
    if d.size < 2:
        return 0.0
    pairs = np.abs(np.diff(d)) / ((d[:-1] + d[1:]) / 2.0)
    return float(100.0 * pairs.mean())


def _boundary_pause_stats(rec: Recording) -> tuple[float, float]:
    """(boundary_pause_ratio, internal_long_pause_count) via reference sentences."""
    if not rec.pauses:
        return 0.0, 0.0
    sent_lens = [len(tokens(s)) for s in rec.reference_sentences]
    boundary_idx = set(np.cumsum(sent_lens) - 1) if sent_lens else set()  # 0-based last word
    n_boundary, n_internal_long = 0, 0
    for p in rec.pauses:
        is_boundary = any(abs(p.after_word_idx - b) <= 1 for b in boundary_idx)
        if is_boundary:
            n_boundary += 1
        elif p.is_long:
            n_internal_long += 1
    return n_boundary / len(rec.pauses), float(n_internal_long)


def extract(rec: Recording, y: np.ndarray, sr: int) -> dict[str, float]:
    g = "prosody"
    feats: dict[str, float] = {}

    # --- Pitch (F0) on voiced frames ---
    # All pitch features are speaker-relative: expressed in semitones vs the utterance's
    # OWN median F0 (within-utterance normalization == the deployable form of per-speaker
    # pitch normalization). Absolute mean pitch (Hz) is deliberately DROPPED -- it is a
    # speaker/gender identity confound, not a proficiency construct, and cannot be
    # per-speaker normalized for an unseen speaker at single-utterance predict time.
    if y.size > 0:
        f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr)
        f0v = f0[~np.isnan(f0)]
    else:
        f0v, voiced = np.array([]), np.array([])
    if f0v.size > 1:
        semitone = 12.0 * np.log2(f0v / np.nanmedian(f0v))
        feats["pitch_sd_semitone"] = float(np.std(semitone))
        feats["pitch_range_semitone"] = float(
            np.percentile(semitone, 95) - np.percentile(semitone, 5)
        )
        feats["voiced_frac"] = float(np.mean(voiced)) if voiced.size else 0.0
    else:
        feats.update(pitch_sd_semitone=0.0, pitch_range_semitone=0.0, voiced_frac=0.0)

    # --- Intensity / energy dynamics ---
    if y.size > 0:
        rms = librosa.feature.rms(y=y)[0]
        rms = rms[rms > 0]
        if rms.size > 1:
            db = 20.0 * np.log10(rms / rms.max())
            feats["energy_sd_db"] = float(np.std(db))
            feats["energy_dynamic_range_db"] = float(np.percentile(db, 95) - np.percentile(db, 5))
        else:
            feats.update(energy_sd_db=0.0, energy_dynamic_range_db=0.0)
    else:
        feats.update(energy_sd_db=0.0, energy_dynamic_range_db=0.0)

    # --- Rhythm: variability of word durations & inter-word intervals ---
    word_durs = [w.end - w.start for w in rec.words]
    interword = [rec.words[i + 1].start - rec.words[i].end for i in range(len(rec.words) - 1)]
    feats["npvi_word_duration"] = _npvi(word_durs)
    feats["npvi_interword"] = _npvi(interword)

    # --- Speaking-rate variability across runs ---
    run_rates = []
    for r in rec.runs:
        if len(r) >= 2:
            span = rec.words[r[-1]].end - rec.words[r[0]].start
            if span > 0:
                run_rates.append(len(r) / span)
    feats["run_rate_sd"] = float(np.std(run_rates, ddof=1)) if len(run_rates) > 1 else 0.0

    # --- Pause placement (boundary vs internal) ---
    b_ratio, internal_long = _boundary_pause_stats(rec)
    feats["boundary_pause_ratio"] = b_ratio
    feats["internal_long_pause_count"] = internal_long

    return {ns(g, k): float(v) for k, v in feats.items()}
