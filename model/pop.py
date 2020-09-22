import click
import pathlib
import pandas as pd
import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"]),
        pd.read_csv(path / 'valid.txt', names=["text"]),
        pd.read_csv(path / 'test.txt', names=["text"]),
    )


def split(x):
    return [(x[:i], x[i]) for i in range(1, len(x) - 1)]


def ev_data(dataset, col="text"):
    dataset["session_id"] = dataset.index
    dataset["splitted"] = dataset[col].apply(split)
    exploded = dataset.explode("splitted")
    exploded["observed"] = exploded["splitted"].str[0]
    exploded["gold"] = exploded["splitted"].str[1]
    return exploded


class PopEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.popualar_ = None

    def fit(self, X, y=None):
        exploded = X.str.split().explode()
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
    model = PopEstimator().fit(train["text"])
    print(ev_data(valid))
    # print(model.predict(valid).shape)


if __name__ == '__main__':
    main()
