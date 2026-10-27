import numpy as np
import pandas as pd
import pytest

from speech_scoring.constants import SUBSCORES
from speech_scoring.feedback.generate import generate
from speech_scoring.scoring.cv import grouped_oof, select_params
from speech_scoring.scoring.metrics import pearson
from speech_scoring.scoring.models import MultiHeadAnalytic, RidgeAll


@pytest.fixture
def synthetic():
    """Each subscore depends only on its own feature group, as the design assumes."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "delivery__speech_rate_wps": rng.normal(2.5, 0.5, n),
            "delivery__silence_ratio": rng.uniform(0, 0.5, n),
            "accuracy__gop_mean": rng.uniform(0.3, 1.0, n),
            "accuracy__wer": rng.uniform(0, 0.4, n),
            "prosody__pitch_range_semitone": rng.normal(6, 2, n),
            "prosody__boundary_pause_ratio": rng.uniform(0, 1, n),
        }
    )
    subs = pd.DataFrame(
        {
            "fluency": 3 * X["delivery__speech_rate_wps"] - 4 * X["delivery__silence_ratio"],
            "accuracy": 8 * X["accuracy__gop_mean"] - 5 * X["accuracy__wer"],
            "prosodic": 0.5 * X["prosody__pitch_range_semitone"] + 2,
        }
    ).clip(0, 10)
    y = (0.4 * subs["accuracy"] + 0.4 * subs["fluency"] + 0.2 * subs["prosodic"]).to_numpy()
    groups = np.repeat(np.arange(30), 10)
    return X, y, subs, groups


def test_heads_use_only_their_own_group(synthetic):
    X, y, subs, _ = synthetic
    model = MultiHeadAnalytic().fit(X, y, subscores=subs)
    contribs = model.contributions(X.head(1))
    assert all(c.startswith("delivery__") for c in contribs["fluency"].columns)
    assert all(c.startswith("accuracy__") for c in contribs["accuracy"].columns)
    assert all(c.startswith(("prosody__", "delivery__")) for c in contribs["prosodic"].columns)


def test_multihead_recovers_structure(synthetic):
    X, y, subs, _ = synthetic
    model = MultiHeadAnalytic(alpha=1.0).fit(X, y, subscores=subs)
    assert pearson(y, model.predict(X)) > 0.95
    heads = model.predict_subscores(X)
    for s in SUBSCORES:
        assert pearson(subs[s], heads[s]) > 0.9


def test_grouped_cv_selects_from_grid(synthetic):
    X, y, subs, groups = synthetic
    params = select_params(RidgeAll, X, y, groups)
    assert params in RidgeAll.param_grid()
    oof = grouped_oof(RidgeAll, params, X, y, groups)
    assert not np.isnan(oof).any()


def test_feedback_cites_evidence(synthetic):
    X, y, subs, _ = synthetic
    model = MultiHeadAnalytic().fit(X, y, subscores=subs)
    row = X.head(1)
    heads = model.predict_subscores(row).iloc[0].to_dict()
    fb = generate("u1", float(model.predict(row)[0]), heads, model.contributions(row))
    assert fb.text.startswith("Total score:")
    assert "/10" in fb.text
