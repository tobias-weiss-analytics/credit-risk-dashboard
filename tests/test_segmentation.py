from credit_risk.segmentation import segment


def test_segmentation(data):
    X, _ = data
    seg = segment(X, ["Credit_Amount", "Duration", "Savings_Account"], k=4)
    assert len(set(seg.labels)) == 4
    assert seg.coords.shape == (1000, 2)
    assert len(set(seg.names.values())) == 4
    assert (seg.assign(X.head(10)) == seg.labels[:10]).all()
