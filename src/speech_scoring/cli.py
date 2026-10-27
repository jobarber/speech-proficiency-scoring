"""`speech-score` command line.

download   fetch speechocean762 (~520 MB)
extract    Whisper + forced alignment (GPU), then interpretable features (CPU)
evaluate   compare models on the held-out speakers; write results + figures
train      fit and save the shipped scorer
score      score the test set from cached features -> outputs/scores_test.csv
predict    score one recording against its prompt -> JSON
explain    plot how one test utterance's score was built
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="speech-score", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download")
    ex = sub.add_parser("extract")
    ex.add_argument("--workers", type=int, default=8)
    ex.add_argument("--skip-align", action="store_true", help="reuse the alignment cache only")
    sub.add_parser("evaluate")
    sub.add_parser("train")
    sub.add_parser("score")
    pr = sub.add_parser("predict")
    pr.add_argument("--audio", required=True, help="16 kHz mono WAV")
    pr.add_argument("--prompt", required=True, help="the sentence the speaker was reading")
    xp = sub.add_parser("explain")
    xp.add_argument("--utt", default=None, help="test utterance id (default: a typical one)")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "download":
        from .data import download

        print(download(settings))
    elif args.cmd == "extract":
        from .features.extract import align, build_features

        if not args.skip_align:
            align(settings)
        print("features:", build_features(settings, workers=args.workers).shape)
    elif args.cmd == "evaluate":
        from .scoring.evaluate import run

        print("wrote", run(settings))
    elif args.cmd == "train":
        from .scoring.final_model import train_and_save

        train_and_save(settings)
        print("saved", settings.output_dir / "model.joblib")
    elif args.cmd == "score":
        from .pipeline import score_corpus

        out = score_corpus(settings)
        print(out.drop(columns="feedback").round(2).head(10).to_string())
        print(f"\n{out.index[0]}: {out['feedback'].iloc[0]}")
    elif args.cmd == "predict":
        from .pipeline import score_audio

        r = score_audio(args.audio, args.prompt, settings)
        print(
            json.dumps(
                {
                    "total": round(r.total, 2),
                    "interval_90": [round(v, 2) for v in r.interval],
                    "subscores": {k: round(v, 2) for k, v in r.subscores.items()},
                    "reliability": r.reliability,
                    "reliability_reasons": r.reliability_reasons,
                    "feedback": r.feedback,
                },
                indent=2,
            )
        )
    elif args.cmd == "explain":
        from .explain import explain_utterance

        print("wrote", explain_utterance(settings, args.utt, Path("assets")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
