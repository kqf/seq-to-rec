import torch
import skorch

from sklearn.base import BaseEstimator, TransformerMixin
from torchtext.data import Example, Dataset, Field


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


def shift(seq, by):
    return torch.cat([seq[by:], seq.new_ones((by, seq.shape[1]))])


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


class SeqNet(skorch.NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        logits = y_pred.view(-1, y_pred.shape[-1])
        return self.criterion_(logits, shift(y_true, by=1).view(-1))


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


def main():
    pass


if __name__ == '__main__':
    main()
