from model.experimental import build_preprocessor, build_model
from torchtext.data import BucketIterator


def test_data(data, batch_size=32):
    dataset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size
    assert batch.text.shape[1] == 5


def test_model(data):
    model = build_model().fit(data)
    preds = model.predict(data)

    # Predict multiple labels
    assert len(preds.shape) == 2
