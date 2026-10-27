"""Feature extraction in two passes.

1. GPU pass (`align`): Whisper word timings + confidences and forced-alignment GOP
   scores per utterance, appended to a JSONL cache so the pass is resumable.
2. CPU pass (`build_features`): delivery, accuracy and prosody features from the
   cache plus the audio, run in parallel worker processes, written to parquet.

Everything downstream (training, evaluation, scoring the corpus) reads the
parquet, so it runs on a laptop CPU without the GPU stack.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..asr import Word
from ..audioio import load_audio
from ..config import Settings, get_settings
from ..data import Utterance, load_utterances
from ..recording import build_recording
from . import accuracy, delivery, gop, prosody


def alignment_cache(settings: Settings) -> Path:
    return settings.output_dir / "alignments.jsonl"


def features_path(settings: Settings) -> Path:
    return settings.output_dir / "features.parquet"


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {r["utt_id"]: r for r in rows}


def align(settings: Settings | None = None, limit: int | None = None) -> Path:
    """Run Whisper + forced alignment on every utterance not yet cached."""
    from ..asr import WhisperTranscriber

    settings = settings or get_settings()
    path = alignment_cache(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_cache(path)
    todo = [u for u in load_utterances(settings) if u.utt_id not in done][:limit]
    whisper = WhisperTranscriber(settings.whisper_model)
    aligner = gop.ForcedAligner()
    with path.open("a", encoding="utf-8") as f:
        for u in tqdm(todo, desc="align"):
            words = whisper.transcribe(u.wav_path)
            scores = aligner.word_scores(u.wav_path, u.text)
            row = {"utt_id": u.utt_id, "words": [asdict(w) for w in words], "gop": scores}
            f.write(json.dumps(row) + "\n")
    return path


def features_for(utt: Utterance, words: list[Word], gop_scores: list[float]) -> dict:
    """All interpretable features for one utterance (also used by `predict`)."""
    rec = build_recording(utt.utt_id, words, utt.text)
    y, sr = load_audio(utt.wav_path)
    row: dict[str, object] = {}
    row.update(delivery.extract(rec, len(y) / sr))
    row.update(accuracy.extract(rec, utt.text))
    row.update(gop.features(gop_scores))
    row.update(prosody.extract(rec, y, sr))
    return row


def _worker(args: tuple[Utterance, list[dict], list[float]]) -> dict:
    utt, words, scores = args
    row = features_for(utt, [Word(**w) for w in words], scores)
    meta = {
        "utt_id": utt.utt_id,
        "speaker": utt.speaker,
        "split": utt.split,
        "age": utt.age,
        "gender": utt.gender,
        "text": utt.text,
    }
    return {**meta, **utt.labels, **row}


def build_features(settings: Settings | None = None, workers: int = 8) -> pd.DataFrame:
    settings = settings or get_settings()
    cache = _read_cache(alignment_cache(settings))
    jobs = [
        (u, cache[u.utt_id]["words"], cache[u.utt_id]["gop"])
        for u in load_utterances(settings)
        if u.utt_id in cache
    ]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(tqdm(pool.map(_worker, jobs, chunksize=16), total=len(jobs), desc="features"))
    df = pd.DataFrame(rows).set_index("utt_id")
    df.to_parquet(features_path(settings))
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "__" in c]


def load_features(settings: Settings | None = None) -> pd.DataFrame:
    path = features_path(settings or get_settings())
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `speech-score extract` first")
    return pd.read_parquet(path)
