import numpy as np
import pytest

from credit_risk.data import PROTECTED_FEATURES
from credit_risk.model import (
    ModelConfig,
    build_pipeline,
    cost_curve,
    decision_metrics,
    expected_cost,
    optimal_threshold,
)


def test_cost_matrix():
    y = np.array([0, 0, 1, 1])
    # one bad approved (5) + one good rejected (1)
    assert expected_cost(y, np.array([1, 0, 0, 1])) == 6
    assert expected_cost(y, np.ones(4, dtype=int)) == 10


def test_optimal_threshold_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = optimal_threshold(y, p)
    assert 0.3 < t <= 0.7
    assert cost_curve(y, p)["cost"].min() == 0


def test_decision_metrics_handles_no_approvals():
    m = decision_metrics(np.array([0, 1]), np.array([0.1, 0.2]), threshold=0.99)
    assert m["approval_rate"] == 0
    assert np.isnan(m["bad_rate_among_approved"])


def test_protected_attributes_not_used(data):
    X, y = data
    pipe = build_pipeline(ModelConfig(n_estimators=20)).fit(X, y)
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    assert not any(n.startswith(tuple(PROTECTED_FEATURES)) for n in names)


def test_trained_model_quality(trained):
    assert trained.metrics["roc_auc"] > 0.70
    assert trained.metrics["cost"] < trained.metrics["cost_approve_all"]
    assert 0.5 < trained.threshold < 1.0
    assert trained.oof_proba.between(0, 1).all()


@pytest.mark.parametrize(
    ("feature", "low", "high"),
    [
        ("Savings_Account", "< 100 DM", ">= 1000 DM"),
        ("Checking_Status", "< 0 DM", ">= 200 DM"),
        ("Employment_Since", "Unemployed", ">= 7 years"),
    ],
)
def test_monotone_constraints(trained, data, feature, low, high):
    """Moving up an ordinal scale must never lower P(good credit)."""
    X, _ = data
    sample = X.sample(100, random_state=0)
    worse, better = sample.assign(**{feature: low}), sample.assign(**{feature: high})
    assert (trained.predict_proba_good(better) >= trained.predict_proba_good(worse) - 1e-9).all()
