import pytest
import pandas as pd


@pytest.fixture
def data(size=320):
    return pd.DataFrame({
        "text": ["1 2 3 4 5", ] * size
    })
