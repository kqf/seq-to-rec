import click
import pathlib
import pandas as pd


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"]),
        pd.read_csv(path / 'valid.txt', names=["text"]),
        pd.read_csv(path / 'test.txt', names=["text"]),
    )


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, test, valid = read_data(path)
    exploded = train.str.split().explode("text")
    freq = exploded.groupby(exploded).size()
    most_popular = freq.sort_values(ascending=False)[:20]
    print(most_popular)


if __name__ == '__main__':
    main()
