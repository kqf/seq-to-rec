import pytest
from torchtext.data import BucketIterator

from model.flat.nn import build_preprocessor
from model.flat.rnn import build_model as rnn
from model.flat.mf import build_model as mf


def test_data(flat_data, batch_size=32):
    dataset = build_preprocessor().fit_transform(flat_data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size
    assert batch.text.shape[1] == 4


@pytest.mark.parametrize("build_model", [
    rnn,
    mf,
])
def test_model(build_model, flat_data):
    model = build_model(k=2).fit(flat_data)
    preds = model.predict(flat_data)

    # Predict only next item labels
    assert len(preds.shape) == 2
