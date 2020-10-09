import click
import torch
import skorch
import random

import numpy as np

from functools import partial
from sklearn.pipeline import make_pipeline

from irmetrics.topk import recall, rr

from model.data import ev_data, read_data
from model.dataset import train_split
from model.evaluation import evaluate, ppx, scoring
from model.flat.nn import SeqNet, DynamicVariablesSetter, inference
from model.flat.nn import build_preprocessor, SequenceIterator

SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class AdditiveAttention(torch.nn.Module):
    def __init__(self, k_dim, q_dim, v_dim):
        super().__init__()
        self._fck = torch.nn.Linear(k_dim, v_dim, bias=False)
        self._fcq = torch.nn.Linear(k_dim, v_dim, bias=True)
        self._fcv = torch.nn.Linear(v_dim, 1)
        self._sig = torch.nn.Sigmoid()

    def forward(self, k, q, v, mask=None):
        energy = self._fcv(self._sig(self._fck(k) + self._fcq(q)))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        p_atten = torch.softmax(energy, dim=1)
        return torch.sum(p_atten * v, dim=1, keepdim=True), p_atten


class Model(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, pad_idx=0, unk_idx=1):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx)
        self._att = AdditiveAttention(emb_dim, emb_dim, emb_dim)
        self._out = torch.nn.Linear(2 * emb_dim, emb_dim)
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def forward(self, inputs, hidden=None):
        mask = self.mask(inputs).unsqueeze(-1)
        embedded = self._emb(inputs) * mask

        sl = embedded[:, [-1], :]
        sg, _ = self._att(embedded, sl, embedded, mask)

        hidden = self._out(torch.cat([sl, sg], dim=-1).squeeze(1))
        return hidden @ self._emb.weight.T

    def mask(self, x):
        return (x != self.pad_idx) & (x != self.unk_idx)


def build_model(X_val=None, k=20):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNet(
        module=Model,
        module__vocab_size=100,  # Dummy dimension
        module__emb_dim=100,
        # module__hidden_dim=100,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.002,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=4,
        batch_size=128,
        iterator_train=SequenceIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=True,
        iterator_train__sort_key=lambda x: len(x.text),
        iterator_valid=SequenceIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=partial(train_split, prep=preprocessor, X_val=X_val),
        device=device,
        predict_nonlinearity=partial(inference, k=k, device=device),
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),  # Original paper
            DynamicVariablesSetter(),
            skorch.callbacks.EpochScoring(
                partial(ppx, entry="valid_loss"),
                name="perplexity",
                use_caching=False,
                lower_is_better=False,
            ),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k, func=recall),
                name="recall@20",
                on_train=False,
                lower_is_better=False,
                use_caching=True
            ),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k, func=rr),
                name="mrr@20",
                on_train=False,
                lower_is_better=False,
                use_caching=True
            ),
            skorch.callbacks.ProgressBar('count'),
        ],
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    data = ev_data(train["text"])

    print(data)
    model = build_model(ev_data(valid["text"])).fit(data)

    evaluate(model, test.sample(frac=0.1), "test")
    evaluate(model, valid, "valid")
    evaluate(model, train, "train")


if __name__ == '__main__':
    main()
