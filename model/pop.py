import click
import pathlib
import pandas as pd
import itertools

from sklearn.base import BaseEstimator, TransformerMixin


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"]),
        pd.read_csv(path / 'valid.txt', names=["text"]),
        pd.read_csv(path / 'test.txt', names=["text"]),
    )


class PopEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.popualar_ = None

    def fit(self, X, y=None):
        exploded = X.str.split().explode("text")
        freq = exploded.groupby(exploded).size()
        self.popualar_ = freq.sort_values(ascending=False)[:20]
        return self

    def predict(self, X):
        return itertools.repeat(self.popualar_, len(X))


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    model = PopEstimator().fit(train)
    print(model.predict(train))


if __name__ == '__main__':
    main()
