"""Model pipeline, cost-sensitive threshold selection and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .data import NOMINAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES, PROTECTED_FEATURES

# Cost matrix from the dataset documentation: approving a bad applicant costs 5,
# rejecting a good applicant costs 1. Positive class = good credit (approve).
COST_APPROVE_BAD = 5
COST_REJECT_GOOD = 1
RANDOM_STATE = 42


class OrdinalWithUnknown(BaseEstimator, TransformerMixin):
    """Encode ordered categories as 0..k-1.

    A category outside the order (e.g. "no checking account") becomes NaN on the
    ordinal scale plus a separate 0/1 indicator column, so a monotone constraint
    on the scale is never forced onto a category that is not part of it.
    """

    def __init__(self, specs: dict[str, tuple[list[str], str | None]]):
        self.specs = specs

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        out = {}
        for col, (order, outside) in self.specs.items():
            out[col] = X[col].map({cat: i for i, cat in enumerate(order)}).astype(float)
            if outside is not None:
                out[f"{col}: {outside}"] = (X[col] == outside).astype(float)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        names = []
        for col, (_, outside) in self.specs.items():
            names.append(col)
            if outside is not None:
                names.append(f"{col}: {outside}")
        return np.asarray(names, dtype=object)


def _onehot_name(feature: str, category) -> str:
    return f"{feature}: {category}"


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    exclude_protected: bool = True
    monotone: bool = True
    test_size: float = 0.2

    def features(self) -> tuple[list[str], dict, list[str]]:
        drop = set(PROTECTED_FEATURES) if self.exclude_protected else set()
        nominal = [f for f in NOMINAL_FEATURES if f not in drop]
        return list(NUMERIC_FEATURES), dict(ORDINAL_FEATURES), nominal


def build_pipeline(config: ModelConfig) -> Pipeline:
    numeric, ordinal, nominal = config.features()
    preprocessor = ColumnTransformer(
        [
            # Trees need no scaling; raw values keep SHAP explanations readable.
            ("num", "passthrough", numeric),
            ("ord", OrdinalWithUnknown(ordinal), list(ordinal)),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  feature_name_combiner=_onehot_name), nominal),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    # Higher value on an ordinal scale may only raise the probability of good credit.
    constraints = {col: 1 for col in ordinal} if config.monotone else None
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        monotone_constraints=constraints,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _cost(fp, fn) -> int:
    return int(fp * COST_APPROVE_BAD + fn * COST_REJECT_GOOD)


def expected_cost(y_true, approve) -> int:
    """Total misclassification cost of approve (1) / reject (0) decisions."""
    tn, fp, fn, tp = confusion_matrix(y_true, approve, labels=[0, 1]).ravel()
    return _cost(fp, fn)


def cost_curve(y_true, proba_good, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.0, 1.0001, 0.01), 2) if thresholds is None else thresholds
    costs = [expected_cost(y_true, (proba_good >= t).astype(int)) for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "cost": costs})


def optimal_threshold(y_true, proba_good) -> float:
    """Threshold on P(good) that minimises total cost (ties -> lowest threshold)."""
    curve = cost_curve(y_true, proba_good)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def decision_metrics(y_true, proba_good, threshold: float) -> dict:
    approve = (np.asarray(proba_good) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, approve, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    cost = _cost(fp, fn)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": cost,
        "cost_per_application": cost / n,
        "accuracy": (tp + tn) / n,
        "approval_rate": (tp + fp) / n,
        "bad_rate_among_approved": fp / (tp + fp) if tp + fp else float("nan"),
        "recall_good": tp / (tp + fn) if tp + fn else float("nan"),
        "recall_bad": tn / (tn + fp) if tn + fp else float("nan"),
    }


@dataclass
class TrainedModel:
    config: ModelConfig
    pipeline: Pipeline
    threshold: float
    X_test: pd.DataFrame
    y_test: pd.Series
    proba_test: np.ndarray
    oof_proba: pd.Series = field(repr=False)
    metrics: dict = field(default_factory=dict)

    def predict_proba_good(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


def train(X: pd.DataFrame, y: pd.Series, config: ModelConfig | None = None) -> TrainedModel:
    """Train, pick the threshold without touching the test set, evaluate on the test set.

    * The threshold is chosen on out-of-fold predictions of the training split.
    * ``oof_proba`` holds out-of-fold scores for every applicant, so the portfolio
      view never shows a score the model produced for its own training rows.
    """
    config = config or ModelConfig()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=RANDOM_STATE
    )
    oof_train = cross_val_predict(build_pipeline(config), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = optimal_threshold(y_train, oof_train)

    pipeline = build_pipeline(config).fit(X_train, y_train)
    proba_test = pipeline.predict_proba(X_test)[:, 1]

    oof_all = cross_val_predict(build_pipeline(config), X, y, cv=cv, method="predict_proba")[:, 1]

    metrics = decision_metrics(y_test, proba_test, threshold)
    metrics["roc_auc"] = roc_auc_score(y_test, proba_test)
    metrics["cost_approve_all"] = expected_cost(y_test, np.ones(len(y_test), dtype=int))
    return TrainedModel(config, pipeline, threshold, X_test, y_test, proba_test,
                        pd.Series(oof_all, index=X.index, name="Score"), metrics)
