import click
import torch
import skorch
import pathlib

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Example, Dataset, Field, BucketIterator


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"]),
        pd.read_csv(path / 'valid.txt', names=["text"]),
        pd.read_csv(path / 'test.txt', names=["text"]),
    )


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fields, min_freq=1):
        self.fields = fields
        self.min_freq = min_freq

    def fit(self, X, y=None):
        dataset = self.transform(X, y)
        for name, field in dataset.fields.items():
            if field.use_vocab:
                field.build_vocab(dataset, min_freq=self.min_freq)
        return self

    def transform(self, X, y=None):
        proc = [X[col].apply(f.preprocess) for col, f in self.fields]
        examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
        return Dataset(examples, self.fields)


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["text"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CollaborativeModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._rnn = torch.nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim)
        self._out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs)
        lstm_out, hidden = self._rnn(embedded, hidden)
        return self._out(lstm_out)


def shift(seq, by):
    """
        Shift the sequence by one
        seq -- tensor[seq_len, batch_size],
        by -- int,
    """
    return torch.cat([seq[by:], seq.new_ones((by, seq.shape[1]))])


class SeqNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, shift(y_true.T, by=1).view(-1))


def tokenize(x):
    return x.split()


def build_preprocessor():
    text_field = Field(
        tokenize=tokenize,
        init_token=None,
        eos_token=None,
        batch_first=True
    )
    fields = [
        ('text', text_field),
    ]
    return TextPreprocessor(fields, min_freq=3)


class SequenceIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.text, batch.text


def ppx(loss_type):
    def _ppx(model, X, y):
        return np.exp(model.history[-1][loss_type])
    _ppx.__name__ = f"ppx_{loss_type}"
    return _ppx


def build_model():
    model = SeqNet(
        module=CollaborativeModel,
        module__vocab_size=100,  # Dummy dimension
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=4,
        batch_size=32,
        iterator_train=SequenceIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SequenceIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=Dataset.split,
        callbacks=[
            skorch.callbacks.GradientNormClipping(1.),
            DynamicVariablesSetter(),
            skorch.callbacks.EpochScoring(ppx("train_loss"), on_train=True),
            skorch.callbacks.EpochScoring(ppx("valid_loss"), on_train=False),
            skorch.callbacks.ProgressBar('count'),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
        model,
    )
    return full


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, valid, test = read_data(path)
    model = build_model().fit(train)
    print(model.predict(valid))


if __name__ == '__main__':
    main()
