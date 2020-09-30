import click
import torch
import skorch
import random
import pathlib
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from torchtext.data import Field, BucketIterator
from torchtext.dataset import TextPreprocessor

from irmetrics.topk import rr, recall

from model.data import ev_data

SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"]),
        pd.read_csv(path / 'test.txt', names=["text"]),
        pd.read_csv(path / 'valid.txt', names=["text"]),
    )


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["text"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')
        print(f'There number of unique items is {len(vocab)}')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RecurrentCollaborativeModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=0.5,
        )
        self._out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs)
        out, hidden = self._rnn(embedded, hidden)
        return self._out(out)[:, -1, :]


class SeqNet(skorch.NeuralNet):
    # def get_loss(self, y_pred, y_true, X=None, training=False):
    #     import ipdb; ipdb.set_trace(); import IPython; IPython.embed() # noqa
    #     logits = y_pred[:, :-1, :].permute(1, 0, 2)
    #     targets = X[:, 1:].T
    #     return self.criterion_(logits, targets.to(logits.device))

    def predict(self, X):
        probas = self.predict_proba(X)
        indexes = (-probas).argsort(-1)[:, :25]
        return np.take(X.fields["text"].vocab.itos, indexes)


def tokenize(x):
    return x.split()


def build_preprocessor(min_freq=5):
    with warnings.catch_warnings(record=True):
        text_field = Field(
            tokenize=tokenize,
            init_token=None,
            eos_token=None,
            batch_first=True,
            pad_first=True,
        )
        fields = [
            ('text', text_field),
            ('gold', text_field),
        ]
        return TextPreprocessor(fields, min_freq=min_freq)


class SequenceIterator(BucketIterator):
    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings(record=True):
            super().__init__(*args, **kwargs)

    def __iter__(self):
        with warnings.catch_warnings(record=True):
            for batch in super().__iter__():
                yield batch.text, batch.gold.view(-1)


def ppx(model, X, y):
    return np.exp(model.history[-1]["train_loss"].item())


def recall_scoring(model, X, y):
    dataset = ev_data(X.sample(frac=0.01)["text"].str.split())
    predicted = model.predict(dataset)
    return np.mean(recall(dataset["gold"], predicted))


def build_model():
    preprocessor = build_preprocessor(min_freq=1)
    model = SeqNet(
        module=RecurrentCollaborativeModel,
        module__vocab_size=100,  # Dummy dimension
        module__emb_dim=100,
        module__hidden_dim=100,
        module__active=torch.nn.Sigmoid(),
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=10,
        batch_size=512,
        iterator_train=SequenceIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=True,
        iterator_train__sort_key=lambda x: len(x.text),
        iterator_valid=SequenceIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        callbacks=[
            skorch.callbacks.GradientNormClipping(150.),  # Original paper
            DynamicVariablesSetter(),
            skorch.callbacks.EpochScoring(
                ppx,
                name="train_perplexity",
                on_train=True,
                use_caching=False,
            ),
            # skorch.callbacks.EpochScoring(
            #     recall_scoring,
            #     name="recall@25",
            #     on_train=True,
            #     use_caching=False
            # ),
            skorch.callbacks.ProgressBar('count'),
        ],
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full


def evaluate(model, data, title):
    dataset = ev_data(data["text"].str.split())
    dataset["text"] = dataset["observed"]

    predicted = model.predict(dataset)
    dataset["recall"] = recall(dataset["gold"], predicted)
    dataset["rr"] = rr(dataset["gold"], predicted)

    print("Evaluating on", title)
    print("Recall", dataset["recall"].mean())
    print("MRR", dataset["rr"].mean())


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    model = build_model().fit(ev_data(train["text"]))

    model[-1].set_params(batch_size=32)
    evaluate(model, train.sample(frac=0.015), "train")
    evaluate(model, valid.sample(frac=0.1), "valid")


if __name__ == '__main__':
    main()