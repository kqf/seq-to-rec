from model.pop import build_model


def test_model(data):
    model = build_model().fit(data)
    preds = model.predict(data)
    assert preds.shape[0] == data["text"].shape[0]
