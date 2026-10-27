from credit_risk.data import CODE_LABELS, FEATURE_NAMES, ORDINAL_FEATURES, sex_from_status


def test_shape_and_target(data):
    X, y = data
    assert X.shape == (1000, 20)
    assert list(X.columns) == FEATURE_NAMES
    assert set(y.unique()) == {0, 1}
    assert y.mean() == 0.7  # 700 good, 300 bad as documented


def test_all_codes_recoded(data):
    X, _ = data
    for col, mapping in CODE_LABELS.items():
        assert set(X[col].unique()) <= set(mapping.values()), col
    assert X.isna().sum().sum() == 0


def test_ordinal_specs_cover_all_labels():
    for col, (order, outside) in ORDINAL_FEATURES.items():
        labels = set(order) | ({outside} if outside else set())
        assert labels == set(CODE_LABELS[col].values()), col


def test_sex_from_status(data):
    X, _ = data
    assert set(sex_from_status(X["Sex_Status"]).unique()) == {"Male", "Female"}
