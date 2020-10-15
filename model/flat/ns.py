import click
import torch
import skorch
import random

import numpy as np

from functools import partial
from sklearn.pipeline import make_pipeline

from irmetrics.topk import recall, rr

from model.data import ev_data, read_data
from model.dataset import train_split
from model.evaluation import evaluate, ppx, scoring
from model.flat.nn import SeqNet, DynamicVariablesSetter, inference
from model.flat.nn import build_preprocessor
from model.flat.quadratic import AdditiveAttention
from torchtext.data import BucketIterator


SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class NegativeSamplingIterator(BucketIterator):
    def __init__(self, dataset, batch_size,
                 neg_samples, ns_exponent, *args, **kwargs):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.ns_exponent = ns_exponent
        self.neg_samples = neg_samples

        vocab = dataset.fields["text"].vocab
        freq = [vocab.freqs[s]**self.ns_exponent for s in vocab.itos]

        # Normalize
        self.freq = np.array(freq) / np.sum(freq)

    def __iter__(self):
        for batch in super().__iter__():
            inputs = {
                "text": batch.text,
                "target": batch.gold,
                "negatives": self.sample(batch.text),
            }
            yield inputs, batch.gold.view(-1)

    def sample(self, text):
        negatives = np.random.choice(
            np.arange(len(self.freq)),
            p=self.freq,
            size=(text.shape[0], self.neg_samples),
        )
        return torch.tensor(negatives, dtype=text.dtype).to(text.device)


class Model(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, pad_idx=0, unk_idx=1):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx)
        self._att = AdditiveAttention(emb_dim, emb_dim, emb_dim)
        self._out = torch.nn.Linear(2 * emb_dim, emb_dim)
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def forward(self, text, target, negatives, hidden=None):
        mask = self.mask(text).unsqueeze(-1)
        embedded = self._emb(text) * mask
        sg, _ = self._att(embedded, embedded, embedded, mask)

        sl = embedded[:, -1, :]
        hidden = self._out(torch.cat([sg, sl], dim=-1))
        return hidden @ self._emb.weight.T

    def mask(self, x):
        return (x != self.pad_idx) & (x != self.unk_idx)


def xavier_init(x):
    if x.dim() < 2:
        return
    return torch.nn.init.xavier_uniform_(x)


def build_model(X_val=None, k=20):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNet(
        module=Model,
        module__vocab_size=100,  # Dummy dimension
        module__emb_dim=100,
        # module__hidden_dim=100,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.002,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=5,
        batch_size=128,
        iterator_train=NegativeSamplingIterator,
        iterator_train__neg_samples=6,
        iterator_train__ns_exponent=3. / 4.,
        iterator_train__shuffle=True,
        iterator_train__sort=True,
        iterator_train__sort_key=lambda x: len(x.text),
        iterator_valid=NegativeSamplingIterator,
        iterator_valid__neg_samples=6,
        iterator_valid__ns_exponent=3. / 4.,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=partial(train_split, prep=preprocessor, X_val=X_val),
        device=device,
        predict_nonlinearity=partial(inference, k=k, device=device),
        callbacks=[
            skorch.callbacks.Initializer("*_fc*", fn=xavier_init),
            skorch.callbacks.GradientNormClipping(1.),  # Original paper
            DynamicVariablesSetter(),
            skorch.callbacks.EpochScoring(
                partial(ppx, entry="valid_loss"),
                name="perplexity",
                use_caching=False,
                lower_is_better=False,
            ),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k, func=recall),
                name="recall@20",
                on_train=False,
                lower_is_better=False,
                use_caching=True
            ),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k, func=rr),
                name="mrr@20",
                on_train=False,
                lower_is_better=False,
                use_caching=True
            ),
            skorch.callbacks.ProgressBar('count'),
        ],
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    data = ev_data(train["text"])

    print(data)
    model = build_model(ev_data(valid["text"])).fit(data)

    evaluate(model, test.sample(frac=0.1), "test")
    evaluate(model, valid, "valid")
    evaluate(model, train, "train")


if __name__ == '__main__':
    main()
