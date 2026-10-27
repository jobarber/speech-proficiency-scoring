"""Labels and the construct map that ties each analytic head to its feature group."""

from __future__ import annotations

# speechocean762 sentence-level scores, 0-10, from five expert raters.
TARGET = "total"
SUBSCORES = ("accuracy", "fluency", "prosodic")

# Each analytic head sees only the feature groups its rubric describes. This is what
# makes the explanation trustworthy: the fluency score can only move because of timing
# and pausing evidence, never because of pitch or pronunciation. The corpus rubric
# defines prosody as intonation *and* stable speed, cadence, stammering and pauses, so
# the prosodic head sees both the prosody and the delivery groups.
HEAD_TO_GROUP = {
    "accuracy": ("accuracy",),
    "fluency": ("delivery",),
    "prosodic": ("prosody", "delivery"),
}

SCORE_LO, SCORE_HI = 0.0, 10.0
GROUP_COL = "speaker"
ID_COL = "utt_id"
