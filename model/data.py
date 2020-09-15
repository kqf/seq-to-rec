import click
import pathlib

import datetime
import numpy as np
import pandas as pd


def read_file(path, filename, frac=None):
    df = pd.read_csv(
        pathlib.Path(path) / filename,
        sep=',',
        header=None,
        names=['session_id', 'time', 'item_id'],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: str, 2: np.int64},
    )
    df["time"] = pd.to_datetime(df["time"])

    if frac is not None:
        df = df.sample(frac=frac)

    # Ensure the data is in the right order
    df = df.sort_values(["session_id", "time"])
    df.reset_index(inplace=True)

    # Remove short sessions
    cleaned = remove_short(df, min_size=1)

    # Keep the items that have at least 4 interactions
    cleaned = remove_short(df, "item_id", min_size=4)
    return cleaned


def remove_short(data, col="session_id", min_size=1):
    lengths = data.groupby(col).size()
    return data[np.in1d(data[col], lengths[lengths > min_size].index)]


def build_sessions(
    df,
    session_col="session_id",
    item_col="item_id",
    min_len=1,
):
    df[item_col] = df[item_col].astype(str)
    sess = df.groupby(session_col)[item_col].transform(lambda x: " ".join(x))
    sess = sess[sess.str.split().str.len() > min_len]
    return sess


@click.command()
@click.option("--raw", type=click.Path(exists=False))
@click.option("--out", type=click.Path(exists=False))
@click.option("--train", default='yoochoose-test.dat')
@click.option("--test", default='yoochoose-test.dat')
def main(raw, out, train, test):
    train = read_file(raw, train, frac=1. / 64.)
    test = read_file(raw, test)

    # Ensure test contains the same ids as the train
    # This is a very doubtful operation!
    test = test[np.in1d(test["item_id"], train["item_id"])]

    # Take the last day for validation
    split_day = train["time"].max() - datetime.timedelta(days=1)
    valid = train[train["time"] >= split_day]
    train = train[train["time"] < split_day]

    train_sessions = build_sessions(train)
    valid_sessions = build_sessions(valid)
    test_sessions = build_sessions(test)

    opath = pathlib.Path(out)
    opath.mkdir(parents=True, exist_ok=False)

    train_sessions.to_csv(opath / "train.txt", index=False, header=None)
    valid_sessions.to_csv(opath / "valid.txt", index=False, header=None)
    test_sessions.to_csv(opath / "test.txt", index=False, header=None)


if __name__ == '__main__':
    main()
