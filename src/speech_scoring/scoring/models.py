"""Scoring models behind one small interface.

All models take a feature DataFrame (columns namespaced `group__name`) so they can
select feature groups by construct and expose per-feature contributions. The
analytic multi-head model is the one that ships; the rest are reference points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..constants import HEAD_TO_GROUP, SCORE_HI, SCORE_LO, SUBSCORES
from ..features.base import FEATURE_SIGNS, features_in_group


def feat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if "__" in c]


class ScoringModel:
    name = "base"
    needs_subscores = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, subscores: pd.DataFrame | None = None):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def param_grid() -> list[dict]:
        return [{}]


class MeanBaseline(ScoringModel):
    name = "baseline_mean"

    def fit(self, X, y, subscores=None):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu_)


@dataclass
class SingleFeature(ScoringModel):
    """OLS on one interpretable feature (e.g. speech rate) -- a strong baseline."""

    feature: str = "delivery__speech_rate_wps"
    name: str = "baseline_wps"

    def fit(self, X, y, subscores=None):
        self.lr_ = LinearRegression().fit(X[[self.feature]].values, y)
        return self

    def predict(self, X):
        return np.clip(self.lr_.predict(X[[self.feature]].values), SCORE_LO, SCORE_HI)


@dataclass
class RidgeAll(ScoringModel):
    """StandardScaler + Ridge on all features -> overall score directly."""

    alpha: float = 10.0
    name: str = "ridge_all"

    def fit(self, X, y, subscores=None):
        self.cols_ = feat_cols(X)
        self.scaler_ = StandardScaler().fit(X[self.cols_].values)
        Z = self.scaler_.transform(X[self.cols_].values)
        self.model_ = Ridge(alpha=self.alpha).fit(Z, y)
        return self

    def predict(self, X):
        Z = self.scaler_.transform(X[self.cols_].values)
        return np.clip(self.model_.predict(Z), SCORE_LO, SCORE_HI)

    def contributions(self, X) -> pd.DataFrame:
        """Per-feature contribution (coef * standardized value) for each row."""
        Z = self.scaler_.transform(X[self.cols_].values)
        contrib = Z * self.model_.coef_
        return pd.DataFrame(contrib, columns=self.cols_, index=X.index)

    @staticmethod
    def param_grid():
        return [{"alpha": a} for a in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)]


@dataclass
class GBMAll(ScoringModel):
    """HistGradientBoosting on all features -- nonlinear, SHAP-explainable."""

    max_depth: int = 2
    learning_rate: float = 0.05
    max_iter: int = 200
    l2_regularization: float = 1.0
    name: str = "gbm_all"

    def fit(self, X, y, subscores=None):
        self.cols_ = feat_cols(X)
        self.model_ = HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            l2_regularization=self.l2_regularization,
            min_samples_leaf=5,
            random_state=0,
        ).fit(X[self.cols_].values, y)
        return self

    def predict(self, X):
        return np.clip(self.model_.predict(X[self.cols_].values), SCORE_LO, SCORE_HI)

    @staticmethod
    def param_grid():
        grid = []
        for depth in (2, 3):
            for lr in (0.03, 0.06):
                grid.append({"max_depth": depth, "learning_rate": lr})
        return grid


class SignedRidge:
    """Ridge regression with per-coefficient sign constraints, via bounded least squares.

    Minimizes ||Zw - (y - mean y)||^2 + alpha ||w||^2 with w_j >= 0, w_j <= 0 or free,
    by stacking sqrt(alpha) * I under Z. Inputs are expected to be standardized.
    """

    def __init__(self, alpha: float, signs: np.ndarray):
        self.alpha = alpha
        self.signs = signs

    def fit(self, Z: np.ndarray, y: np.ndarray):
        self.intercept_ = float(np.mean(y))
        A = np.vstack([Z, np.sqrt(self.alpha) * np.eye(Z.shape[1])])
        b = np.concatenate([y - self.intercept_, np.zeros(Z.shape[1])])
        lo = np.where(self.signs > 0, 0.0, -np.inf)
        hi = np.where(self.signs < 0, 0.0, np.inf)
        self.coef_ = lsq_linear(A, b, bounds=(lo, hi)).x
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.coef_ + self.intercept_


@dataclass
class MultiHeadAnalytic(ScoringModel):
    """The shipped, explainable model.

    One ridge head per construct predicts its subscore from the feature groups its
    rubric describes (see constants.HEAD_TO_GROUP). A linear
    combiner maps the three predicted subscores to the total score. The explanation
    is the model itself: per-feature contributions within each head, plus the
    combiner's construct weights.

    Each head is sign-constrained (features.base.FEATURE_SIGNS), so correlated features
    cannot take opposite signs and contradict each other in an explanation.
    """

    alpha: float = 10.0
    name: str = "multihead"
    needs_subscores: bool = True
    heads_: dict = field(default_factory=dict)

    def fit(self, X, y, subscores=None):
        assert subscores is not None, "MultiHeadAnalytic requires subscores"
        self.heads_ = {}
        head_preds = {}
        for sub, groups in HEAD_TO_GROUP.items():
            cols = [c for g in groups for c in features_in_group(feat_cols(X), g)]
            scaler = StandardScaler().fit(X[cols].values)
            Z = scaler.transform(X[cols].values)
            signs = np.array([FEATURE_SIGNS.get(c, 0) for c in cols])
            ridge = SignedRidge(self.alpha, signs).fit(Z, subscores[sub].values)
            self.heads_[sub] = {"cols": cols, "scaler": scaler, "ridge": ridge, "groups": groups}
            head_preds[sub] = ridge.predict(Z)
        H = np.column_stack([head_preds[s] for s in SUBSCORES])
        # Non-negative weights: a better subscore can never lower the total. The ridge
        # penalty splits weight between heads that share features (fluency and prosody
        # both see timing) instead of arbitrarily handing it all to one of them.
        self.combiner_ = Ridge(alpha=10.0, positive=True).fit(H, y)
        return self

    def _head_matrix(self, X):
        preds = {}
        for sub, h in self.heads_.items():
            Z = h["scaler"].transform(X[h["cols"]].values)
            preds[sub] = h["ridge"].predict(Z)
        return preds, np.column_stack([preds[s] for s in SUBSCORES])

    def predict(self, X):
        _, H = self._head_matrix(X)
        return np.clip(self.combiner_.predict(H), SCORE_LO, SCORE_HI)

    def predict_subscores(self, X) -> pd.DataFrame:
        preds, _ = self._head_matrix(X)
        # Clip so out-of-distribution input (e.g. the wrong prompt) stays on the scale.
        return pd.DataFrame(preds, index=X.index).clip(SCORE_LO, SCORE_HI)

    def combiner_weights(self) -> dict[str, float]:
        return {s: float(w) for s, w in zip(SUBSCORES, self.combiner_.coef_)}

    def contributions(self, X) -> dict[str, pd.DataFrame]:
        """Per-head per-feature contributions (coef * standardized value)."""
        out = {}
        for sub, h in self.heads_.items():
            Z = h["scaler"].transform(X[h["cols"]].values)
            out[sub] = pd.DataFrame(Z * h["ridge"].coef_, columns=h["cols"], index=X.index)
        return out

    @staticmethod
    def param_grid():
        return [{"alpha": a} for a in (1.0, 3.0, 10.0, 30.0, 100.0)]


FEATURE_MODELS = {
    "baseline_mean": MeanBaseline,
    "baseline_wps": SingleFeature,
    "ridge_all": RidgeAll,
    "gbm_all": GBMAll,
    "multihead": MultiHeadAnalytic,
}
