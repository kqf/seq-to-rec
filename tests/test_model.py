import pytest
import pandas as pd

from model.model import build_preprocessor, build_model
from torchtext.data import BucketIterator


@pytest.fixture
def data(size=320):
    return pd.DataFrame({
        "text": ["1 2 3 4 5", ] * size
    })


def test_data(data, batch_size=32):
    dataset = build_preprocessor().fit_transform(data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size
    assert batch.text.shape[1] == 5


def test_model(data):
    model = build_model().fit(data)
    model.predict(data)
