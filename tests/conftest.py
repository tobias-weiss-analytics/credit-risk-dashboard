import pytest

from credit_risk.data import load_german_credit
from credit_risk.model import ModelConfig, train


@pytest.fixture(scope="session")
def data():
    return load_german_credit()


@pytest.fixture(scope="session")
def trained(data):
    X, y = data
    return train(X, y, ModelConfig(n_estimators=150))
