"""Text normalization and edit-distance metrics shared across the pipeline.

Used by the accuracy features (WER/CER against the prompt).
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation (keep intra-word apostrophes), collapse space."""
    t = str(text).lower().replace("’", "'")
    t = _WORD_RE.sub(" ", t)
    return " ".join(t.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def _levenshtein_counts(ref: list, hyp: list) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via DP alignment."""
    n, m = len(ref), len(hyp)
    # cost matrix + backtrace
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    # backtrace to count S/D/I
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dele, ins


def error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    """Token-level WER + char-level CER with S/D/I breakdown (all rate per ref).

    Self-contained (no external dep) so it is deterministic and unit-testable.
    """
    ref_t, hyp_t = tokens(reference), tokens(hypothesis)
    s, d, i = _levenshtein_counts(ref_t, hyp_t)
    n = max(1, len(ref_t))
    ref_c = list(normalize(reference).replace(" ", ""))
    hyp_c = list(normalize(hypothesis).replace(" ", ""))
    cs, cd, ci = _levenshtein_counts(ref_c, hyp_c)
    nc = max(1, len(ref_c))
    return {
        "wer": (s + d + i) / n,
        "sub_rate": s / n,
        "del_rate": d / n,
        "ins_rate": i / n,
        "cer": (cs + cd + ci) / nc,
        "ref_word_count": float(len(ref_t)),
        "hyp_word_count": float(len(hyp_t)),
    }
