import torch
import skorch
import warnings

import numpy as np

from torchtext.data import Field, BucketIterator


from model.dataset import TextPreprocessor


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["text"].vocab
        net.set_params(module__vocab_size=len(vocab))
        net.set_params(module__pad_idx=vocab["<pad>"])
        net.set_params(module__unk_idx=vocab["<unk>"])
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')
        print(f'There number of unique items is {len(vocab)}')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SeqNet(skorch.NeuralNet):
    def predict(self, X):
        # Now predict_proba returns top k indexes
        indexes = self.predict_proba(X)
        return np.take(X.fields["text"].vocab.itos, indexes)


def build_preprocessor(min_freq=5):
    with warnings.catch_warnings(record=True):
        text_field = Field(
            tokenize=None,
            init_token=None,
            pad_token="<unk>",
            unk_token="<unk>",
            eos_token=None,
            batch_first=True,
            pad_first=True,
        )
        fields = [
            ('text', text_field),
            ('gold', text_field),
        ]
        return TextPreprocessor(fields, min_freq=min_freq)


class SequenceIterator(BucketIterator):
    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings(record=True):
            super().__init__(*args, **kwargs)

    def __iter__(self):
        with warnings.catch_warnings(record=True):
            for batch in super().__iter__():
                yield batch.text, batch.gold.view(-1)


def inference(logits, k, device):
    probas = torch.softmax(logits.to(device), dim=-1)
    # Return only indices
    return torch.topk(probas, k=k, dim=-1)[-1].clone().detach()
