import pytest
import pandas as pd

from model.pop import PopEstimator


@pytest.fixture
def data(size=320):
    return pd.DataFrame({
        "text": ["1 2 3 4 5", ] * size
    })


def test_model(data):
    model = PopEstimator().fit(data["text"].str.split())
    preds = model.predict(data)
    assert preds.shape[0] == data["text"].shape[0]
