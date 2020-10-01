from model.data import ev_data
from model.flat import build_preprocessor, build_model
from torchtext.data import BucketIterator


def test_data(data, batch_size=32):
    dataset = build_preprocessor().fit_transform(ev_data(data["text"]))
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size
    assert batch.text.shape[1] == 4


def test_model(data):
    model = build_model().fit(ev_data(data["text"]))
    preds = model.predict(ev_data(data["text"]))

    # Predict only next item labels
    assert len(preds.shape) == 2
