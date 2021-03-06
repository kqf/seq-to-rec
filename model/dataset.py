import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Example, Dataset


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
        with warnings.catch_warnings(record=True):
            proc = [X[col].apply(f.preprocess) for col, f in self.fields]
            examples = [Example.fromlist(f, self.fields) for f in zip(*proc)]
            return Dataset(examples, self.fields)


def train_split(X, prep, X_val):
    if X_val is None:
        return Dataset.split(X)
    return X, prep.transform(X_val)
