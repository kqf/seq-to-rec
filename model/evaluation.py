from irmetrics.topk import recall, rr

from model.timer import timer


def evaluate(model, data, title):
    with timer("Predict"):
        predicted = model.predict(data)

    with timer("Calculate the vectorized measures"):
        data["recall"] = recall(data["gold"], predicted)
        data["rr"] = rr(data["gold"], predicted)

    print("Evaluating on", title)
    print("Recall", data["recall"].mean())
    print("MRR", data["rr"].mean())
    return data
