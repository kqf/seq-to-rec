import click
import pathlib

import numpy as np
import pandas as pd


def read_file(path, filename):
    df = pd.read_csv(
        pathlib.Path(path) / filename,
        sep=',',
        header=None,
        names=['session_id', 'time', 'item_id'],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: str, 2: np.int64},
    )
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = df["time"].apply(lambda x: x.timestamp())

    # Remove short sessions
    cleaned = remove_short(df, min_size=1)

    # Keep the items that have at least 4 interactions
    cleaned = remove_short(df, "item_id", min_size=4)
    return cleaned


def remove_short(data, col="session_id", min_size=1):
    lengths = data.groupby(col).size()
    return data[np.in1d(data[col], lengths[lengths > min_size].index)]


@click.command()
@click.option("--raw", type=click.Path(exists=False))
@click.option("--out", type=click.Path(exists=False))
@click.option("--train", default='yoochoose-test.dat')
@click.option("--test", default='yoochoose-test.dat')
def main(raw, out, train, test):
    train = read_file(raw, train)
    test = read_file(raw, test)
    print(train)


if __name__ == '__main__':
    main()
