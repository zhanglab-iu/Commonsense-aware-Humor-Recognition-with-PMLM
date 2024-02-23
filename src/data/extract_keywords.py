import os
from argparse import ArgumentParser

import pandas as pd
import yake
from joblib import Parallel, delayed


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--algo", type=str, default="seqm")
    parser.add_argument("--window_size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--column", type=str, default="text")

    return parser


def load_dataset(input_file: str) -> pd.DataFrame:
    compression = None
    if input_file.endswith("zip"):
        compression = "zip"
    elif input_file.endswith("gz"):
        compression = "gzip"

    actual_filename = input_file
    if compression == "gzip":
        actual_filename = input_file.removesuffix(".gz")
    elif compression == "zip":
        actual_filename = input_file.removesuffix(".zip")

    sep = "\t" if actual_filename.endswith("tsv") else ","

    df = pd.read_csv(
        input_file,
        compression=compression,
        index_col=False,
        # on_bad_lines="warn",
        lineterminator="\n",
        sep=sep,
    )
    return df


def save_dataset(data: pd.DataFrame, output_file: os.PathLike) -> None:
    data.to_csv(
        output_file,
        header=True,
        index=False,
        sep="\t",
        line_terminator="\n",
        compression="gzip",
    )
    return


def extract_keywords(extractor: yake.KeywordExtractor, text: str) -> list:
    keywords = extractor.extract_keywords(text=text)
    keywords = [keyword[0] for keyword in keywords]
    return keywords


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    verbose = args.verbose

    # load dataset
    dataset = load_dataset(args.input_file)
    print(dataset.head())

    kw_extractor = yake.KeywordExtractor(
        lan=args.lang,
        n=args.ngram,
        dedupLim=args.threshold,
        dedupFunc=args.algo,
        windowsSize=args.window_size,
        top=args.topk,
    )

    col = args.column
    texts = dataset[col]
    print("Data size:", len(texts))

    keywords = Parallel(n_jobs=-1, verbose=1)(
        delayed(extract_keywords)(kw_extractor, text) for text in texts
    )

    output_file = args.output_file
    if output_file is None:
        filename = args.input_file.split("/")[-1]
        output_file = f"data/processed/{filename}"

    dataset["keywords"] = keywords

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_dataset(data=dataset, output_file=output_file)
