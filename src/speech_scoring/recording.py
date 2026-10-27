"""Build a Recording: words + pauses + runs + pause-annotated transcript.

This is the shared intermediate that every feature extractor consumes. Pause
thresholds follow fluency research (~0.25-0.3s is the standard silent-pause floor).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .asr import Word

# Silent-pause thresholds (seconds).
PAUSE_MIN = 0.30  # below this = normal coarticulation gap, not a pause
PAUSE_LONG = 1.00  # >= this = long pause
RUN_BREAK = 0.30  # gap that ends a "run" (for mean length of run)

FILLED_PAUSE_RE = re.compile(r"^(uh+|um+|er+|hmm+|mm+)$", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Pause:
    start: float
    end: float
    after_word_idx: int  # pause occurs after words[after_word_idx]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_long(self) -> bool:
        return self.duration >= PAUSE_LONG


@dataclass
class Recording:
    sample_id: str
    words: list[Word]
    reference_text: str
    pauses: list[Pause] = field(default_factory=list)
    runs: list[list[int]] = field(default_factory=list)  # each run = list of word indices

    @property
    def reference_sentences(self) -> list[str]:
        return [s.strip() for s in _SENT_SPLIT.split(self.reference_text.strip()) if s.strip()]

    @property
    def speech_start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def speech_end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def phonation_time(self) -> float:
        """Total time inside words (sum of word durations)."""
        return sum(w.end - w.start for w in self.words)

    def annotated_transcript(self) -> str:
        """Verbatim word stream with <pause_short>/<pause_long>/<filled> tokens."""
        if not self.words:
            return ""
        out: list[str] = []
        for i, w in enumerate(self.words):
            tok = w.word.strip()
            if FILLED_PAUSE_RE.match(re.sub(r"[^a-zA-Z]", "", tok)):
                out.append("<filled>")
            else:
                out.append(tok)
            if i + 1 < len(self.words):
                gap = self.words[i + 1].start - w.end
                if gap >= PAUSE_LONG:
                    out.append("<pause_long>")
                elif gap >= PAUSE_MIN:
                    out.append("<pause_short>")
        return " ".join(out)


def _detect_pauses(words: list[Word]) -> list[Pause]:
    pauses = []
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap >= PAUSE_MIN:
            pauses.append(Pause(start=words[i].end, end=words[i + 1].start, after_word_idx=i))
    return pauses


def _detect_runs(words: list[Word]) -> list[list[int]]:
    """Group word indices into runs separated by gaps >= RUN_BREAK."""
    if not words:
        return []
    runs, cur = [], [0]
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if gap >= RUN_BREAK:
            runs.append(cur)
            cur = []
        cur.append(i + 1)
    if cur:
        runs.append(cur)
    return runs


def build_recording(sample_id: str, words: list[Word], reference_text: str) -> Recording:
    rec = Recording(sample_id=sample_id, words=list(words), reference_text=reference_text)
    rec.pauses = _detect_pauses(rec.words)
    rec.runs = _detect_runs(rec.words)
    return rec
