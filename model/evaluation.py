import numpy as np

from irmetrics.topk import recall, rr

from model.timer import timer
from model.data import ev_data


def evaluate(model, dataset, title):
    data = ev_data(dataset["text"])

    with timer("Predict"):
        predicted = model.predict(data)

    with timer("Calculate the vectorized measures"):
        data["recall"] = recall(data["gold"], predicted)
        data["rr"] = rr(data["gold"], predicted)

    print("Evaluating on", title)
    print("Recall", data["recall"].mean())
    print("MRR", data["rr"].mean())
    return data


def scoring(model, X, y, k, func):
    return func(y, model.predict_proba(X), k=k).mean()


def ppx(model, X, y, entry="train_loss"):
    loss = model.history[-1, entry]
    return np.exp(-loss.item())
