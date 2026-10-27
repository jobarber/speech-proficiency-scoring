"""Speaker-grouped cross-validation.

A speaker never appears on both sides of a split. Utterances from one speaker share
voice, accent and recording conditions, so mixing them across folds overstates how
well a model handles new learners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import pearson
from .models import ScoringModel


def fit_model(model_cls, params: dict, X, y, subscores=None) -> ScoringModel:
    model = model_cls(**params)
    model.fit(X, y, subscores=subscores if model.needs_subscores else None)
    return model


def grouped_oof(
    model_cls,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    subscores: pd.DataFrame | None = None,
    n_splits: int = 5,
) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sub = subscores.iloc[tr] if subscores is not None else None
        oof[te] = fit_model(model_cls, params, X.iloc[tr], y[tr], sub).predict(X.iloc[te])
    return oof


def select_params(model_cls, X, y, groups, subscores=None) -> dict:
    """Pick hyperparameters by grouped-CV Pearson on the training speakers only."""
    grid = model_cls.param_grid()
    if len(grid) == 1:
        return grid[0]
    scores = [pearson(y, grouped_oof(model_cls, p, X, y, groups, subscores)) for p in grid]
    return grid[int(np.nanargmax(scores))]
