import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from credit_risk.data import load_german_credit  # noqa: E402
from credit_risk.model import ModelConfig, train  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return load_german_credit()


@pytest.fixture(scope="session")
def trained(data):
    X, y = data
    return train(X, y, ModelConfig(n_estimators=150))
