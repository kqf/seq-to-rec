import click
import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from irmetrics.topk import recall, rr


from model.data import ev_data, read_data
from model.timer import timer


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
        model = build_model().fit()

    evaluate(model, valid, "validatoin")
    evaluate(model, test, "test")


if __name__ == '__main__':
    main()
