import click
import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from model.timer import timer
from model.data import read_data, ev_data
from model.evaluation import evaluate


class SplitSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def transform(self, X):
        return X[self.col].str.split()

    def fit(self, X, y=None):
        return self


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


def build_model(col="text"):
    model = make_pipeline(
        SplitSelector(col=col),
        PopEstimator(),
    )
    return model


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)

    with timer("Fit the data"):
        model = build_model().fit(train)

    evaluate(model, ev_data(valid["text"]), "validatoin")
    evaluate(model, ev_data(test["text"]), "test")


if __name__ == '__main__':
    main()
