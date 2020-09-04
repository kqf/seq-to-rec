import pytest
import pandas as pd


@pytest.fixture
def data(size=10):
    return pd.DataFrame({
        "text": [[1, 2, 3, 4, 5], ] * size
    })


def test_data(data):
    print(data)
