"""Train and persist the shipped scorer (multi-head model + interval width)."""

from __future__ import annotations

from dataclasses import dataclass

import joblib

from ..config import Settings, get_settings
from ..constants import GROUP_COL, SUBSCORES, TARGET
from ..features.extract import feature_columns, load_features
from .confidence import conformal_halfwidth
from .cv import fit_model, grouped_oof, select_params
from .models import MultiHeadAnalytic


@dataclass
class ScorerBundle:
    model: MultiHeadAnalytic
    conformal_q: float
    subscore_stats: dict[str, tuple[float, float]]
    feature_columns: list[str]


def model_path(settings: Settings):
    return settings.output_dir / "model.joblib"


def train_and_save(settings: Settings | None = None) -> ScorerBundle:
    """Fit on the official training speakers, so the test set stays untouched."""
    settings = settings or get_settings()
    df = load_features(settings)
    tr = df[df["split"] == "train"]
    cols = feature_columns(df)
    X, y, groups, subs = (
        tr[cols],
        tr[TARGET].to_numpy(),
        tr[GROUP_COL].to_numpy(),
        tr[list(SUBSCORES)],
    )
    params = select_params(MultiHeadAnalytic, X, y, groups, subs)
    oof = grouped_oof(MultiHeadAnalytic, params, X, y, groups, subs)
    bundle = ScorerBundle(
        model=fit_model(MultiHeadAnalytic, params, X, y, subs),
        conformal_q=conformal_halfwidth(y, oof, alpha=0.1),
        subscore_stats={s: (float(tr[s].mean()), float(tr[s].std())) for s in SUBSCORES},
        feature_columns=cols,
    )
    path = model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_model(settings: Settings | None = None) -> ScorerBundle:
    settings = settings or get_settings()
    path = model_path(settings)
    return joblib.load(path) if path.is_file() else train_and_save(settings)
