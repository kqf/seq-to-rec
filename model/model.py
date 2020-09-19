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
        # net.set_params(criterion__ignore_index=vocab["<pad>"])

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
        self._rnn = torch.nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim)
        self._out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs)
        lstm_out, hidden = self._rnn(embedded, hidden)
        return self._out(lstm_out)


class CollaborativeModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=128):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, emb_dim)
        self._fc0 = torch.nn.Linear(emb_dim, hidden_dim)
        self._out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs)

        # average over seq dimension
        n = torch.arange(embedded.shape[1], device=embedded.device) + 1
        cmean = torch.cumsum(embedded, dim=1) / n[None, :, None]

        hidden = self._fc0(cmean)
        return self._out(hidden)


class SampledCriterion(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, targets):
        batch_size = targets.shape[-1]
        y = targets.repeat(1, batch_size).view(-1, batch_size)
        logits = logits.reshape(-1, logits.shape[-1])
        return self.criterion(torch.take(logits, y))


class FlattenCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, logits, targets):
        n_outputs = logits.shape[-1]
        return super().forward(
            logits.reshape(-1, n_outputs),
            targets.reshape(-1)
        )


def batch_diagonal_idx(x):
    return torch.arange(x.shape[-1]).repeat(x.shape[0] // x.shape[-1])


def batch_diagonal(x):
    return x[:, batch_diagonal_idx(x)].diag()


class BPRLoss(torch.nn.Module):
    def forward(self, logits):
        diag = batch_diagonal(logits)
        diff = diag.view(-1, 1) - logits
        losses = torch.nn.functional.logsigmoid(diff).mean(dim=-1)
        return -losses.mean()


class Top1Loss(torch.nn.Module):
    def forward(self, logits):
        diag = batch_diagonal(logits).view(-1, 1)
        diff = diag.view(-1, 1) - logits
        losses = (
            torch.nn.functional.sigmoid(-diff) +  # noqa
            torch.nn.functional.sigmoid(logits ** 2)).mean(dim=-1)
        return -losses.mean()


class UnsupervisedCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, logits):
        # Assuming the logits is square matrics, with true answers on diagonal
        return super().forward(logits, batch_diagonal_idx(logits))


class SeqNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred[:, :-1, :].permute(1, 0, 2)
        targets = X[:, 1:].T
        return self.criterion_(logits, targets.to(logits.device))

    def transform(self, X, at=20):
        self.module_.eval()
        xpreds, labels = [], []
        pad_idx = X.fields["text"].vocab.stoi["<pad>"]
        unk_idx = X.fields["text"].vocab.stoi["<unk>"]
        for (x, _) in self.get_iterator(self.get_dataset(X), training=False):
            with torch.no_grad():
                preds = self.module_(x.to(self.device)).detach()

                # Don't generate candidate for the last item in the sequence
                candidates = (-preds).argsort(-1)[:, :-1, :at]

                # Flatten the data
                candidates = candidates.reshape(-1, candidates.shape[-1])
                candidates = candidates.cpu().numpy()

            true_labels = x[:, 1:].reshape(-1, 1).detach().cpu().numpy()

            idx = ((true_labels != pad_idx) | (true_labels != unk_idx))
            idx = idx.reshape(-1)

            xpreds.append(candidates[idx]), labels.append(true_labels[idx])
        return np.vstack(xpreds), np.vstack(labels)

    def predict_proba(self, X):
        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            # Take the last element of the sequence
            y_probas.append(skorch.utils.to_numpy(yp[:, -1, :]))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def predict(self, X):
        probas = self.predict_proba(X)
        indexes = (-probas).argsort()
        return np.take(X.fields["text"].vocab.itos, indexes)


def tokenize(x):
    return x.split()


def build_preprocessor(min_freq=5):
    text_field = Field(
        tokenize=tokenize,
        init_token=None,
        eos_token=None,
        batch_first=True
    )
    fields = [
        ('text', text_field),
    ]
    return TextPreprocessor(fields, min_freq=min_freq)


class SequenceIterator(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.text, None


def ppx(model, X, y):
    return np.exp(model.history[-1]["train_loss"].item())


def recall(y_true, y_pred=None, ignore=None, k=20):
    y_pred = y_pred[:k]
    mask = ~(y_pred == ignore)
    relevant = np.in1d(y_pred[:k], y_true)
    return np.sum(relevant * mask) / y_true[:k].shape[0]


def rr(y_true, y_pred, k=20):
    relevant = np.in1d(y_pred[:k], y_true)
    if not relevant.any():
        return 0
    index = relevant.argmax()
    return relevant[index] / (index + 1)


def scoring(model, X, y, at=20):
    preds, gold = model.transform(X.sample(frac=0.01), at=at)
    return np.mean([recall(g, p) for g, p in zip(gold, preds)])


def build_model():
    model = SeqNet(
        module=CollaborativeModel,
        module__vocab_size=100,  # Dummy dimension
        optimizer=torch.optim.Adam,
        criterion=FlattenCrossEntropy,
        max_epochs=10,
        batch_size=64,
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
            # skorch.callbacks.GradientNormClipping(1.),
            DynamicVariablesSetter(),
            # skorch.callbacks.EpochScoring(
            #     ppx, name="train_perplexity", on_train=True),
            # skorch.callbacks.EpochScoring(
            #     scoring, name="recall@25", on_train=True),
            skorch.callbacks.ProgressBar('count'),
        ],
    )

    full = make_pipeline(
        build_preprocessor(min_freq=1),
        model,
    )
    return full


def evaluate(model, data, title):
    splitted = data["text"].str.split()
    data["text"] = splitted.str[:-1]
    gold = splitted.str[-1].values.reshape(-1, 1)

    preds = model.predict(data)
    recalls = [recall(g, p) for g, p in zip(gold, preds)]
    mean_recall = np.mean(recalls)

    print()
    print(f"{title} recall@25 {mean_recall:.4g}")
    mrr = np.mean([rr(g, p) for g, p in zip(gold, preds)])
    print(f"{title}    mrr@25 {mrr:.4g}")


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, valid, test = read_data(path)
    model = build_model().fit(train)

    model[-1].set_params(batch_size=32)
    evaluate(model, train.sample(frac=0.015), "train")
    evaluate(model, valid.sample(frac=0.5), "valid")


if __name__ == '__main__':
    main()
