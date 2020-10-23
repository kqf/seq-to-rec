import click
import torch
import skorch
import random
import warnings

import numpy as np

from functools import partial
from sklearn.pipeline import make_pipeline

from irmetrics.topk import recall, rr

from model.data import ev_data, read_data
from model.dataset import train_split
from model.evaluation import evaluate, ppx, scoring
from model.flat.nn import SeqNet, inference
from model.flat.nn import build_preprocessor, SequenceIterator
from model.flat.quadratic import AdditiveAttention
from torchtext.data import BucketIterator


SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["text"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(module__pad_idx=vocab["<pad>"])
        net.set_params(module__unk_idx=vocab["<unk>"])
        # Don't ignore any indeces for criterion
        # net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')
        print(f'There number of unique items is {len(vocab)}')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NegativeSamplingIterator(BucketIterator):
    def __init__(self, dataset, batch_size,
                 neg_samples, ns_exponent, *args, **kwargs):
        with warnings.catch_warnings(record=True):
            super().__init__(dataset, batch_size, *args, **kwargs)
        self.ns_exponent = ns_exponent
        self.neg_samples = neg_samples

        vocab = dataset.fields["text"].vocab
        freq = [vocab.freqs[s]**self.ns_exponent for s in vocab.itos]

        # Ensure no "pad/unk" as negative samples
        freq[vocab.stoi["<unk>"]] = 0
        freq[vocab.stoi["<pad>"]] = 0

        # Normalize
        self.freq = np.array(freq) / np.sum(freq)

    def sample(self, text, gold=None):
        negatives = np.random.choice(
            np.arange(len(self.freq)),
            # p=self.freq,
            size=(text.shape[0], self.neg_samples),
        )
        return torch.tensor(negatives, dtype=text.dtype).to(text.device)


class ExampleNegativeSamplingIterator(NegativeSamplingIterator):
    def __iter__(self):
        with warnings.catch_warnings(record=True):
            for batch in super().__iter__():
                samples = self.sample(batch.text)
                negatives = torch.cat([batch.gold, samples], dim=-1)
                inputs = {
                    "text": batch.text,
                    "negatives": negatives,
                }
                yield inputs, torch.arange(
                    samples.shape[1], device=batch.gold.device)

    def sample(self, text):
        negatives = np.random.choice(
            np.arange(len(self.freq)),
            # p=self.freq,
            size=(text.shape[0], self.neg_samples),
        )
        return torch.tensor(negatives, dtype=text.dtype).to(text.device)


class BatchNegativeSamplingIterator(NegativeSamplingIterator):
    def __iter__(self):
        with warnings.catch_warnings(record=True):
            for batch in super().__iter__():
                inputs = {
                    "text": batch.text,
                    "negatives": batch.gold,
                }
                yield inputs, torch.arange(
                    batch.gold.shape[0], device=batch.gold.device)


class FlattenNegativeSamplingIterator(NegativeSamplingIterator):
    def __iter__(self):
        with warnings.catch_warnings(record=True):
            for batch in super().__iter__():
                samples = self.sample(batch.text, batch.gold)
                negatives = torch.cat([batch.gold, samples], dim=-1)
                inputs = {
                    "text": batch.text,
                    "negatives": negatives,
                }
                yield inputs, negatives.shape[-1] * torch.arange(
                    batch.gold.shape[0], device=batch.gold.device)

    def sample(self, text, gold=None):
        freq = np.array(self.freq, copy=True)
        freq[gold] = 0

        negatives = np.random.choice(
            np.arange(len(self.freq)),
            p=self.freq,
            size=(text.shape[0], self.neg_samples),
        )

        negatives = torch.randint(
            (1, len(self.freq),)
            (self.neg_samples,),
            device=text.device
        )
        return negatives
        # return torch.tensor(negatives, dtype=text.dtype).to(text.device)


class Model(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, pad_idx=0, unk_idx=1):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_idx)
        self._att = AdditiveAttention(emb_dim, emb_dim, emb_dim)
        self._out = torch.nn.Linear(2 * emb_dim, emb_dim)
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def forward(self, text, negatives=None, hidden=None):
        mask = self.mask(text).unsqueeze(-1)
        embedded = self._emb(text) * mask
        sg, _ = self._att(embedded, embedded, embedded, mask)

        sl = embedded[:, -1, :]
        hidden = self._out(torch.cat([sg, sl], dim=-1))

        if negatives is not None:
            negatives = negatives.view(-1)

        return hidden @ self._emb.weight[negatives].squeeze().T

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
        iterator_train=FlattenNegativeSamplingIterator,
        # iterator_train=SequenceIterator,
        iterator_train__neg_samples=200,
        iterator_train__ns_exponent=0.,
        iterator_train__shuffle=True,
        # iterator_train__sort=False,
        # iterator_train__sort_key=lambda x: len(x.text),
        iterator_valid=SequenceIterator,
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
