from speech_scoring.asr import Word
from speech_scoring.recording import build_recording


def _words():
    # two words, big gap (1.2s), two more words -> one long pause, two runs
    return [
        Word("the", 0.0, 0.3, 0.99),
        Word("cat", 0.35, 0.6, 0.98),
        Word("sat", 1.8, 2.1, 0.97),  # gap of 1.2s before this
        Word("down", 2.15, 2.4, 0.96),
    ]


def test_pause_and_run_detection():
    rec = build_recording("s", _words(), "The cat sat down.")
    assert len(rec.pauses) == 1
    assert rec.pauses[0].is_long
    assert len(rec.runs) == 2


def test_annotated_transcript_has_pause_token():
    rec = build_recording("s", _words(), "The cat sat down.")
    ann = rec.annotated_transcript()
    assert "<pause_long>" in ann
    assert ann.split()[0] == "the"


def test_empty_recording():
    rec = build_recording("s", [], "ref")
    assert rec.annotated_transcript() == ""
    assert rec.phonation_time == 0.0
