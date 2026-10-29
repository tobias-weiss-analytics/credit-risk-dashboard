from credit_risk.segmentation import segment


def test_segmentation(data):
    X, _ = data
    seg = segment(X, ["Credit_Amount", "Duration", "Savings_Account"], k=4)
    assert len(set(seg.labels)) == 4
    assert seg.coords.shape == (1000, 2)
    assert len(set(seg.names.values())) == 4
    assert (seg.assign(X.head(10)) == seg.labels[:10]).all()


def test_single_feature_names_are_informative(data):
    X, _ = data
    seg = segment(X, ["Age"], k=6)
    names = list(seg.names.values())
    assert len(set(names)) == 6
    # Names that share a level carry the centroid instead of a bare counter.
    assert not any(n.endswith(("(2)", "(3)", "(4)")) for n in names)
