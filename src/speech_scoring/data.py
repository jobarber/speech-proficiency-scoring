"""speechocean762: 5,000 read-aloud English sentences from 250 Mandarin L1 speakers.

Each utterance has sentence-level accuracy, fluency, prosodic and total scores
(0-10) from five experts. The official split is speaker-disjoint (125 / 125).
Zhang et al., 2021. CC BY 4.0. https://www.openslr.org/101/
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Settings, get_settings
from .constants import SUBSCORES, TARGET

URL = "https://www.openslr.org/resources/101/speechocean762.tar.gz"


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    speaker: str
    split: str
    wav_path: Path
    text: str
    labels: dict[str, float]
    age: int = -1
    gender: str = ""


def download(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    root = settings.corpus_dir
    if (root / "resource" / "scores.json").is_file():
        return root
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    archive = settings.data_dir / "speechocean762.tar.gz"
    if not archive.is_file():
        print(f"downloading {URL} (~520 MB)")
        urllib.request.urlretrieve(URL, archive)  # noqa: S310 - fixed OpenSLR URL
    with tarfile.open(archive) as t:
        t.extractall(settings.data_dir, filter="data")
    archive.unlink()
    return root


def _kv(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return dict(line.split(maxsplit=1) for line in lines if line.strip())


def load_utterances(settings: Settings | None = None) -> list[Utterance]:
    settings = settings or get_settings()
    root = settings.corpus_dir
    scores = json.loads((root / "resource" / "scores.json").read_text(encoding="utf-8"))
    out = []
    for split in ("train", "test"):
        wavs = _kv(root / split / "wav.scp")
        spk = _kv(root / split / "utt2spk")
        age = _kv(root / split / "spk2age")
        gender = _kv(root / split / "spk2gender")
        for utt, rel in sorted(wavs.items()):
            s = scores[utt]
            speaker = spk[utt].strip()
            out.append(
                Utterance(
                    utt_id=utt,
                    speaker=speaker,
                    split=split,
                    wav_path=root / rel.strip(),
                    text=s["text"],
                    labels={k: float(s[k]) for k in (*SUBSCORES, TARGET)},
                    age=int(age[speaker]),
                    gender=gender[speaker].strip(),
                )
            )
    return out


def metadata(settings: Settings | None = None) -> pd.DataFrame:
    rows = [
        {
            "utt_id": u.utt_id,
            "speaker": u.speaker,
            "split": u.split,
            "age": u.age,
            "gender": u.gender,
            "text": u.text,
            **u.labels,
        }
        for u in load_utterances(settings)
    ]
    return pd.DataFrame(rows).set_index("utt_id")


def rater_scores(settings: Settings | None = None) -> dict[str, dict[str, list[float]]]:
    """Per-utterance, per-label lists of the five experts' individual scores."""
    settings = settings or get_settings()
    path = settings.corpus_dir / "resource" / "scores-detail.json"
    detail = json.loads(path.read_text(encoding="utf-8"))
    return {utt: {k: v[k] for k in (*SUBSCORES, TARGET)} for utt, v in detail.items()}
