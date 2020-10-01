from model.flat import build_preprocessor, build_model
from torchtext.data import BucketIterator


def test_data(flat_data, batch_size=32):
    dataset = build_preprocessor().fit_transform(flat_data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size
    assert batch.text.shape[1] == 4


def test_model(flat_data):
    model = build_model().fit(flat_data)
    preds = model.predict(flat_data)

    # Predict only next item labels
    assert len(preds.shape) == 2
