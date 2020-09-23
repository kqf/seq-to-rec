import time
import click
import pathlib
import pandas as pd
import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{color}[{name}] done in {et:.0f} s{nocolor}".format(
        name=name, et=time.time() - t0,
        color='\033[1;33m', nocolor='\033[0m'))


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"])["text"].str.split(),
        pd.read_csv(path / 'valid.txt', names=["text"])["text"].str.split(),
        pd.read_csv(path / 'test.txt', names=["text"])["text"].str.split(),
    )


def split(x):
    return [(x[:i + 1], x[i + 1]) for i in range(len(x) - 1)]


def recall(y_true, y_pred=None, ignore=None, k=20):
    y_true = np.atleast_1d(y_true)
    y_pred = y_pred[:k]
    mask = ~(y_pred == ignore)
    relevant = np.in1d(y_pred[:k], y_true)
    return np.sum(relevant * mask) / y_true[:k].shape[0]


def precall(y_true, y_pred=None, ignore=None, k=20):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)
    y_true = y_true.T[:, :k]
    y_pred = y_pred[:, :k]

    relevant = (y_true == y_pred).any(-1) / y_true.shape[-1]
    recalls = np.squeeze(relevant)
    if not recalls.shape:
        return recalls.item()
    return recalls


def ev_data(dataset):
    dataset = dataset[dataset.str.len() > 1].reset_index(drop=True)
    data = pd.DataFrame({"session_id": dataset.index})
    data["splitted"] = dataset.apply(split)
    exploded = data.explode("splitted")
    exploded["observed"] = exploded["splitted"].str[0]
    exploded["gold"] = exploded["splitted"].str[1]
    return exploded.drop(columns=["splitted"])


class PopEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.popualar_ = None

    def fit(self, X, y=None):
        exploded = X.explode()
        freq = exploded.groupby(exploded).size()
        popular = freq.sort_values(ascending=False)[:20].index.values
        self.popualar_ = popular
        return self

    def predict(self, X):
        preds = list(itertools.repeat(self.popualar_, len(X)))
        return np.stack(preds)


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    with timer("Fit the data"):
        model = PopEstimator().fit(train)

    with timer("Prepare the evaluation"):
        ev_valid = ev_data(valid)

    with timer("Predict"):
        predicted = model.predict(ev_valid)

    with timer("Calculate the measures"):
        ev_valid["recall"] = [
            recall(g, p) for g, p in zip(ev_valid["gold"], predicted)]

    with timer("Calculate the vectorized measures"):
        ev_valid["precall"] = precall(ev_valid["gold"], predicted)

    print(ev_valid["precall"].mean() - ev_valid["precall"].mean())
    # print(precall(1, [1, 2, 3, 4]))


if __name__ == '__main__':
    main()
