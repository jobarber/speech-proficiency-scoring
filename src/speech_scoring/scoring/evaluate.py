"""Model comparison and construct-validity checks on the official speaker-disjoint split.

Hyperparameters are chosen by speaker-grouped CV on the training speakers; every
number reported here is on the 125 held-out test speakers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..constants import GROUP_COL, HEAD_TO_GROUP, SUBSCORES, TARGET
from ..data import rater_scores
from ..features.base import GROUPS, features_in_group
from ..features.extract import feature_columns, load_features
from .confidence import conformal_halfwidth
from .cv import fit_model, grouped_oof, select_params
from .metrics import all_metrics, pearson
from .models import GBMAll, MeanBaseline, MultiHeadAnalytic, RidgeAll, SingleFeature

ARMS = {
    "Mean": MeanBaseline,
    "Speech rate only": SingleFeature,
    "Ridge, all features": RidgeAll,
    "Gradient boosting, all features": GBMAll,
    "Multi-head (shipped)": MultiHeadAnalytic,
}
# Sentence-level PCC reported for GOPT (Gong et al., ICASSP 2022), a transformer
# trained on phone-level GOP features with all labels. A strong, non-interpretable
# reference point on the same split.
GOPT = {"accuracy": 0.714, "fluency": 0.753, "prosodic": 0.760, "total": 0.742}

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GRAY = "#1f4e8c", "#d9822b", "#9aa5b1"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)


def split(df: pd.DataFrame):
    cols = feature_columns(df)
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    return tr, te, cols


def speaker_bootstrap(y, p, groups, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI for Pearson, resampling speakers (not utterances)."""
    rng = np.random.default_rng(seed)
    speakers = np.unique(groups)
    idx_by = {s: np.where(groups == s)[0] for s in speakers}
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_by[s] for s in rng.choice(speakers, len(speakers))])
        vals.append(pearson(y[idx], p[idx]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def human_ceiling(utt_ids, raters: dict) -> dict[str, float]:
    """Pearson of each expert against the mean of the other four, averaged over experts.

    This is the agreement a single extra human rater would reach, so it is a fair
    reference for a model predicting the consensus score.
    """
    out = {}
    for label in (*SUBSCORES, TARGET):
        mat = np.array([raters[u][label] for u in utt_ids], dtype=float)  # (n, 5)
        rs = []
        for k in range(mat.shape[1]):
            others = np.delete(mat, k, axis=1).mean(1)
            rs.append(pearson(mat[:, k], others))
        out[label] = float(np.mean(rs))
    return out


def subgroup_table(te: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Mean difference (model - human) per group, in units of the human score SD."""
    sd = te[TARGET].std()
    df = te.assign(pred=pred, group=np.where(te["age"] < 16, "child (<16)", "adult (16+)"))
    rows = {}
    for col in ("group", "gender"):
        for g, part in df.groupby(col):
            rows[g] = {
                "n": len(part),
                "human_mean": part[TARGET].mean(),
                "model_mean": part["pred"].mean(),
                "pearson": pearson(part[TARGET].to_numpy(), part["pred"].to_numpy()),
                "smd": (part["pred"].mean() - part[TARGET].mean()) / sd,
            }
    return pd.DataFrame(rows).T


def compare_models(tr, te, cols) -> pd.DataFrame:
    X_tr, X_te = tr[cols], te[cols]
    y_tr, y_te = tr[TARGET].to_numpy(), te[TARGET].to_numpy()
    g_tr, g_te = tr[GROUP_COL].to_numpy(), te[GROUP_COL].to_numpy()
    sub_tr = tr[list(SUBSCORES)]
    rows = {}
    for name, cls in ARMS.items():
        params = select_params(cls, X_tr, y_tr, g_tr, sub_tr)
        pred = fit_model(cls, params, X_tr, y_tr, sub_tr).predict(X_te)
        m = all_metrics(y_te, pred)
        lo, hi = speaker_bootstrap(y_te, pred, g_te) if name != "Mean" else (np.nan, np.nan)
        rows[name] = {**m, "pearson_lo": lo, "pearson_hi": hi}
    return pd.DataFrame(rows).T


def construct_matrix(tr, te, cols) -> pd.DataFrame:
    """Test Pearson of a ridge on each feature group alone, against each human label."""
    out = {}
    for group in (*GROUPS, "all"):
        gcols = cols if group == "all" else features_in_group(cols, group)
        out[group] = {}
        for label in (*SUBSCORES, TARGET):
            model = RidgeAll(alpha=10.0).fit(tr[gcols], tr[label].to_numpy())
            out[group][label] = pearson(te[label].to_numpy(), model.predict(te[gcols]))
    return pd.DataFrame(out).T


def fig_heads(head_r: dict[str, float], human: dict[str, float], out: Path) -> None:
    labels = [*SUBSCORES, TARGET]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=160)
    ours = [head_r[k] for k in labels]
    ax.bar(x - 0.19, ours, 0.36, color=BLUE, label="this model (interpretable)")
    ax.bar(x + 0.19, [GOPT[k] for k in labels], 0.36, color=GRAY, label="GOPT (published)")
    for i, v in enumerate(ours):
        ax.text(i - 0.19, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    for i, k in enumerate(labels):
        ax.plot(
            [i - 0.4, i + 0.4],
            [human[k]] * 2,
            color=ORANGE,
            lw=1.5,
            label="one expert vs. the other four" if i == 0 else None,
        )
    ax.set_xticks(x, [k.capitalize() for k in labels])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Pearson r, test speakers", fontsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_matrix(matrix: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=160)
    ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=0.85, aspect="auto")
    for i, g in enumerate(matrix.index):
        for j, lab in enumerate(matrix.columns):
            v = matrix.loc[g, lab]
            diag = g in HEAD_TO_GROUP.get(lab, ())
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.5 else INK,
                fontweight="bold" if diag else "normal",
            )
    ax.set_xticks(range(matrix.shape[1]), [c.capitalize() for c in matrix.columns], fontsize=8)
    ax.set_yticks(range(matrix.shape[0]), [f"{g} features" for g in matrix.index], fontsize=8)
    ax.set_xlabel("human label", fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_scatter(y, p, q: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.4), dpi=160)
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(y))
    ax.scatter(y + jitter, p, s=3, alpha=0.25, color=BLUE, lw=0)
    grid = np.linspace(0, 10, 50)
    ax.plot(grid, grid, color=MUTED, lw=0.8)
    ax.fill_between(grid, grid - q, grid + q, color=ORANGE, alpha=0.12, lw=0)
    covered = np.mean(np.abs(y - p) <= q)
    ax.text(
        0.3,
        9.4,
        f"90% interval: ±{q:.1f}\ncoverage on test: {covered:.0%}",
        fontsize=7,
        color=INK,
        va="top",
    )
    ax.set_xlim(0, 10.3)
    ax.set_ylim(0, 10.3)
    ax.set_xlabel("human total score", fontsize=8)
    ax.set_ylabel("model total score", fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run(settings: Settings | None = None, assets: Path = Path("assets")) -> Path:
    settings = settings or get_settings()
    assets.mkdir(parents=True, exist_ok=True)
    df = load_features(settings)
    tr, te, cols = split(df)

    comparison = compare_models(tr, te, cols)
    matrix = construct_matrix(tr, te, cols)

    # The shipped model: heads vs their own human labels, plus the combined total.
    g_tr = tr[GROUP_COL].to_numpy()
    sub_tr = tr[list(SUBSCORES)]
    params = select_params(MultiHeadAnalytic, tr[cols], tr[TARGET].to_numpy(), g_tr, sub_tr)
    model = fit_model(MultiHeadAnalytic, params, tr[cols], tr[TARGET].to_numpy(), sub_tr)
    heads = model.predict_subscores(te[cols])
    total = model.predict(te[cols])
    head_r = {s: pearson(te[s].to_numpy(), heads[s].to_numpy()) for s in SUBSCORES}
    head_r[TARGET] = pearson(te[TARGET].to_numpy(), total)

    oof = grouped_oof(MultiHeadAnalytic, params, tr[cols], tr[TARGET].to_numpy(), g_tr, sub_tr)
    q = conformal_halfwidth(tr[TARGET].to_numpy(), oof, alpha=0.1)

    human = human_ceiling(te.index, rater_scores(settings))
    fig_heads(head_r, human, assets / "heads_vs_gopt.png")
    fig_matrix(matrix, assets / "construct_matrix.png")
    fig_scatter(te[TARGET].to_numpy(), total, q, assets / "total_scatter.png")

    weights = model.combiner_weights()
    md = [
        f"# Results (speechocean762 test set: {len(te)} utterances, "
        f"{te[GROUP_COL].nunique()} unseen speakers)\n",
        "## Total score, by model\n",
        comparison[
            ["pearson", "pearson_lo", "pearson_hi", "qwk", "exact_agree", "adjacent_agree", "rmse"]
        ]
        .round(3)
        .to_markdown(),
        "\n## Shipped model, each head against its own human label\n",
        pd.DataFrame(
            {"this model": head_r, "GOPT (published)": GOPT, "one expert vs. others": human}
        )
        .round(3)
        .to_markdown(),
        "\n## Construct validity: each feature group alone (ridge), Pearson on test\n",
        matrix.round(3).to_markdown(),
        "\n## Subgroups (total score, test set)\n",
        subgroup_table(te, total).round(3).to_markdown(),
        "\n## Combiner weights (total = sum of weight x head score + intercept)\n",
        pd.Series(weights, name="weight").round(3).to_frame().to_markdown(),
        f"\nConformal 90% half-width (from grouped out-of-fold residuals): {q:.2f}. "
        f"Empirical test coverage: {np.mean(np.abs(te[TARGET] - total) <= q):.3f}.",
    ]
    out = settings.output_dir / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    (settings.output_dir / "results.json").write_text(
        json.dumps(
            {
                "heads": head_r,
                "human": human,
                "comparison": comparison.round(4).to_dict(),
                "matrix": matrix.round(4).to_dict(),
                "conformal_q": q,
                "test_coverage": float(np.mean(np.abs(te[TARGET] - total) <= q)),
            },
            indent=2,
        )
    )
    return out
