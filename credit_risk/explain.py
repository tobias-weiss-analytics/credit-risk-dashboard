"""SHAP explanations with human-readable feature values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .data import ORDINAL_FEATURES
from .model import TrainedModel


def make_explainer(trained: TrainedModel, background: pd.DataFrame, n_background: int = 100) -> shap.TreeExplainer:
    pre = trained.pipeline.named_steps["preprocessor"]
    bg = pre.transform(background.sample(min(n_background, len(background)), random_state=42))
    return shap.TreeExplainer(trained.pipeline.named_steps["model"], data=bg, model_output="probability")


def explain(trained: TrainedModel, explainer: shap.TreeExplainer, X: pd.DataFrame, readable: bool = True) -> shap.Explanation:
    """SHAP values on the probability of *good* credit for the rows of ``X``.

    With ``readable=True`` the displayed feature values are the original labels
    (e.g. "< 100 DM" instead of the encoded rank), which suits waterfall plots.
    """
    transformed = trained.pipeline.named_steps["preprocessor"].transform(X)
    exp = explainer(transformed)
    exp.feature_names = list(transformed.columns)
    if readable:
        display = transformed.astype(object).copy()
        for col in transformed.columns:
            if col in ORDINAL_FEATURES:
                display[col] = X[col].values
            elif set(np.unique(transformed[col].dropna())) <= {0.0, 1.0} and col not in X.columns:
                display[col] = np.where(transformed[col] == 1, "yes", "no")
        exp.data = display.to_numpy()
    return exp
