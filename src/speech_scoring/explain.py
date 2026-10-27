"""Figure: how one utterance's total score is assembled from heads and features."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Settings
from .constants import SUBSCORES, TARGET
from .features.extract import load_features
from .feedback.generate import FEATURE_PHRASES, SUBSCORE_LABELS, generate
from .scoring.final_model import load_model

BLUE, RED, INK, MUTED = "#1f4e8c", "#c23b22", "#1f2933", "#7b8794"


def _typical_utterance(df, model, cols) -> str:
    """A test utterance in the middle of the score range with a confident prediction."""
    te = df[df["split"] == "test"]
    pred = model.predict(te[cols])
    err = np.abs(pred - te[TARGET].to_numpy())
    mid = (te[TARGET] >= 5) & (te[TARGET] <= 7) & (err < 0.5)
    return te[mid].index[0]


def explain_utterance(settings: Settings, utt: str | None, assets: Path) -> Path:
    bundle = load_model(settings)
    model, cols = bundle.model, bundle.feature_columns
    df = load_features(settings)
    utt = utt or _typical_utterance(df, model, cols)
    X = df.loc[[utt], cols]
    heads = model.predict_subscores(X).iloc[0]
    contribs = model.contributions(X)
    weights = model.combiner_weights()
    total = float(model.predict(X)[0])

    fig = plt.figure(figsize=(10, 3.8), dpi=160)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=1.0)

    # Waterfall: intercept, then each head's weighted contribution, ending at the total.
    ax = fig.add_subplot(gs[0])
    intercept = float(model.combiner_.intercept_)
    steps = [("intercept", intercept)] + [
        (f"{s.capitalize()}  {heads[s]:.1f} × {weights[s]:.2f}", weights[s] * heads[s])
        for s in SUBSCORES
    ]
    left = 0.0
    for i, (_label, v) in enumerate(steps):
        ax.barh(i, v, left=left, color="#cbd2d9" if i == 0 else BLUE, height=0.6)
        left += v
    ax.axvline(total, color=INK, lw=1)
    ax.set_yticks(range(len(steps)), [label for label, _ in steps], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"total {total:.1f}  (human {df.loc[utt, TARGET]:.0f})", fontsize=9, color=INK)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    for k, s in enumerate(SUBSCORES):
        ax = fig.add_subplot(gs[k + 1])
        row = contribs[s].iloc[0]
        row = row[row.index.isin(FEATURE_PHRASES)].sort_values(key=abs).tail(5)
        names = [textwrap.fill(FEATURE_PHRASES[f][0], 22) for f in row.index]
        ax.barh(range(len(row)), row.to_numpy(), color=[BLUE if v > 0 else RED for v in row])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(row)), names, fontsize=6.5)
        ax.set_title(f"{SUBSCORE_LABELS[s].split(' (')[0]}: {heads[s]:.1f}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(
        0.5,
        0.01,
        "Feature contributions in score points (blue raises, red lowers)",
        ha="center",
        fontsize=7,
        color=MUTED,
    )
    out = assets / "explanation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    fb = generate(utt, total, heads.to_dict(), contribs, bundle.subscore_stats)
    print(f'{utt}: "{df.loc[utt, "text"] if "text" in df else ""}"\n{fb.text}')
    return out
