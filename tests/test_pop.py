from model.pop import PopEstimator


def test_model(data):
    model = PopEstimator().fit(data["text"].str.split())
    preds = model.predict(data)
    assert preds.shape[0] == data["text"].shape[0]
