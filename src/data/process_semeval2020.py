import re
from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False)


def process(df: pd.DataFrame) -> pd.DataFrame:
    df_org = df.copy()
    df_edited = df.copy()

    # preprocess
    df_org["original"] = df_org["original"].apply(lambda x: re.sub(r"\t", "", x))
    df_edited["original"] = df_edited["original"].apply(lambda x: re.sub(r"\t", "", x))

    # original headlines
    df_org["text"] = df_org["original"].apply(lambda x: re.sub(r"\<|\/\>", "", x))
    df_org["score"] = -1

    # edited
    df_edited["text"] = df_edited[["original", "edit"]].apply(
        lambda x: re.sub(r"\<.+\/\>", x[1], x[0]), axis=1
    )
    df_edited["score"] = df_edited["meanGrade"]

    df = pd.concat([df_edited, df_org])
    return df.drop_duplicates("text")  # avoid data duplication on df_org


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=False,
        default="data/raw/semeval-2020-task-7-dataset",
        help="data directory of semeval 2020 task 7",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="data/processed/semeval-2020-task-7-dataset",
        help="output directory for processed data",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) / "subtask-1"
    output_dir = Path(args.output_dir) / "subtask-1"

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_dir / "train.csv")
    df = process(df)
    print(df.head())
    df.to_csv(
        output_dir / "train.tsv.gz",
        compression="gzip",
        sep="\t",
        line_terminator="\n",
        header=True,
        index=False,
    )

    df = load_dataset(input_dir / "dev.csv")
    df = process(df)
    df.to_csv(
        output_dir / "dev.tsv.gz",
        compression="gzip",
        sep="\t",
        line_terminator="\n",
        header=True,
        index=False,
    )

    df = load_dataset(input_dir, "test.csv")
    df = process(df)
    df.to_csv(
        output_dir / "test.tsv.gz",
        compression="gzip",
        sep="\t",
        line_terminator="\n",
        header=True,
        index=False,
    )
