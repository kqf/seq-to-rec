import pytest
from torchtext.data import BucketIterator

from model.flat.nn import build_preprocessor
from model.flat.srnn import build_model as srnn
from model.flat.rnn import build_model as rnn
from model.flat.mf import build_model as mf
from model.flat.srgnn import build_model as srgnn
from model.flat.quadratic import build_model as quadratic
from model.flat.ns import build_model as ns
from model.flat.bpr import build_model as bpr
from model.flat.bns import build_model as bns


def test_data(flat_data, flat_oov, batch_size=32):
    model = build_preprocessor().fit(flat_data)
    dataset = model.transform(flat_data)
    batch = next(iter(BucketIterator(dataset, batch_size=batch_size)))
    assert batch.text.shape[0] == batch_size
    assert batch.text.shape[1] == 4
    oov = model.transform(flat_oov)

    oov_batch = next(iter(BucketIterator(oov, batch_size=batch_size)))
    assert oov_batch.text.shape == batch.text.shape
    assert (oov_batch.text == 0).all()


@pytest.mark.parametrize("build_model", [
    rnn,
    mf,
    srnn,
    srgnn,
    quadratic,
    ns,
    bpr,
    bns,
])
def test_model(build_model, flat_data):
    model = build_model(k=2).fit(flat_data)
    preds = model.predict(flat_data)

    # Predict only next item labels
    assert len(preds.shape) == 2
