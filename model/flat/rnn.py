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


class RecurrentCollaborativeModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128,
                 pad_idx=0, unk_idx=1):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=0.5,
        )
        self._out = torch.nn.Linear(hidden_dim, vocab_size)
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs) * self.mask(inputs).unsqueeze(-1)
        out, hidden = self._rnn(embedded, hidden)
        return self._out(out)[:, -1, :]

    def mask(self, x):
        return (x != self.pad_idx) & (x != self.unk_idx)


def build_model(X_val=None, k=20):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNet(
        module=RecurrentCollaborativeModel,
        module__vocab_size=100,  # Dummy dimension
        module__emb_dim=100,
        module__hidden_dim=100,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.002,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=1,
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

    model[-1].set_params(batch_size=32)

    evaluate(model, test, "test")
    evaluate(model, valid, "valid")
    evaluate(model, train, "train")


if __name__ == '__main__':
    main()
