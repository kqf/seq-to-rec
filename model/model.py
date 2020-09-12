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
        self._fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self._out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self._emb(inputs)
        hidden = self._fc0(embedded)
        hidden = self._fc1(hidden)
        return self._out(hidden)


class SampledCriterion(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, targets):
        losses = []
        for candidates, sampled in zip(logits, targets):
            losses.append(self.criterion(candidates[:, sampled]))
        return torch.mean(torch.stack(losses))


class FlattenCriterion(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, targets):
        n_outputs = logits.shape[-1]
        return self.criterion(
            logits.reshape(-1, n_outputs),
            targets.reshape(-1)
        )


class BPRLoss(torch.nn.Module):
    def forward(self, logits):
        diff = logits.diag().view(-1, 1) - logits
        loss = -torch.mean(torch.nn.functional.logsigmoid(diff))
        return loss


class UnsupervisedCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, logits):
        # Assuming the logits is square matrics, with true answers on diagonal
        labels = torch.arange(logits.shape[-1], device=logits.device)
        return super().forward(logits, labels)


class SeqNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred[:, :-1, :].permute(1, 0, 2)
        targets = X[:, 1:].T.to(self.device)
        return self.criterion_(logits, targets)

    def transform(self, X, at=20):
        self.module_.eval()
        xpreds, labels = [], []
        for (x, _) in self.get_iterator(self.get_dataset(X), training=False):
            with torch.no_grad():
                preds = self.module_(x.to(self.device))

                # Don't generate candidate for the last item in the sequence
                candidates = (-preds).argsort(-1)[:, :-1, :at]

                # Flatten the data
                candidates = candidates.reshape(-1, candidates.shape[-1])
                candidates = candidates.detach().cpu().numpy()

            true_labels = x[:, 1:].reshape(-1, 1).detach().cpu().numpy()
            xpreds.append(candidates), labels.append(true_labels)
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
            yield batch.text, None


def ppx(model, X, y):
    return np.exp(model.history[-1]["train_loss"].item())


def recall(y_true, y_pred=None, ignore=None, k=25):
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


def scoring(model, X, y, at=25):
    preds, gold = model.transform(X, at=at)
    return np.mean([recall(g, p) for g, p in zip(gold, preds)])


def build_model():
    model = SeqNet(
        module=CollaborativeModel,
        module__vocab_size=100,  # Dummy dimension
        optimizer=torch.optim.Adam,
        criterion=SampledCriterion,
        criterion__criterion=BPRLoss(),
        max_epochs=2,
        batch_size=32,
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
            skorch.callbacks.GradientNormClipping(1.),
            DynamicVariablesSetter(),
            skorch.callbacks.EpochScoring(
                ppx, name="train_perplexity", on_train=True),
            skorch.callbacks.EpochScoring(
                scoring, name="recall@25", on_train=True),
            skorch.callbacks.ProgressBar('count'),
        ],
    )

    full = make_pipeline(
        build_preprocessor(),
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
    train = train[train["text"].str.split().str.len() < 125]
    model = build_model().fit(train)

    evaluate(model, train, "train")
    evaluate(model, valid, "valid")


if __name__ == '__main__':
    main()
