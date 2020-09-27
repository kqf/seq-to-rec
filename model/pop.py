import time
import click
import pathlib
import pandas as pd
import itertools

import numpy as np
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
from irmetrics.topk import recall, rr


from model.data import ev_data


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


def evaluate(model, data, title):
    with timer("Prepare the evaluation"):
        dataset = ev_data(data)

    with timer("Predict"):
        predicted = model.predict(dataset)

    with timer("Calculate the vectorized measures"):
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
    with timer("Fit the data"):
        model = PopEstimator().fit(train)
    evaluate(model, valid, "validatoin")
    evaluate(model, test, "test")


if __name__ == '__main__':
    main()
