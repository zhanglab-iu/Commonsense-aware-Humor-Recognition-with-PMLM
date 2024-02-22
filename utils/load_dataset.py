import ast
import sys
from pathlib import Path
from typing import Optional, Union

import imblearn
import numpy as np
import pandas as pd

if sys.version_info > (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

__all__ = [
    "load_dataset",
    "TASK",
    "PHASE",
]

TASK = Literal[
    "HaHackathon",
    "rJokes",
    "ColBERT",
    "Humicroedit",
]
PHASE = Literal["train", "valid", "test", "dev"]


def load_dataset(
    *,
    data_dir: Union[str, Path],
    task: TASK,
    phase: PHASE = "train",
    score: Optional[int] = None,
    describe: bool = True,
    debug: bool = False,
    bs: Optional[int] = None,
    seed: int = 42,
    undersampling: bool = False,
) -> pd.DataFrame:
    # Assertion check following Literal below
    assert phase in ["train", "valid", "test", "dev"]
    if score is not None:
        assert 0 <= score <= 10

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if phase == "valid":
        phase = "dev"

    # build path
    filename = phase + ".tsv.gz"

    dataset = pd.read_csv(
        data_dir / filename,
        compression="gzip",
        # index_col=0 if task == "HaHackathon" else False,
        index_col=False,
        on_bad_lines="warn",
        sep="\t",
        lineterminator="\n",
    )

    if task == "HaHackathon":
        dataset = dataset.rename(columns={"is_humor": "label"})  # for HaHackathon
    elif task == "ColBERT":
        dataset = dataset.rename(columns={"humor": "label"})
    elif task == "Humicroedit":
        dataset = dataset.rename(columns={"score": "label"})
        dataset = dataset[dataset["label"] != -1]

        dataset["label"][dataset["label"] < 1] = 0
        dataset["label"][dataset["label"] >= 1] = 1
        dataset = dataset.drop_duplicates("text")  # data cleaning

    if undersampling and phase == "train":  # undersampling
        sampler = imblearn.under_sampling.RandomUnderSampler(
            sampling_strategy="majority", random_state=seed
        )
        dataset, _ = sampler.fit_resample(dataset, dataset["label"])
    dataset["keywords"] = dataset["keywords"].apply(
        ast.literal_eval
    )  # "['a', 'b']" to ['a', 'b']

    if debug:
        if bs is None:
            bs = 2
        print(f"Fetch {bs} instance from dataset")
        dataset = dataset.iloc[np.random.choice(len(dataset), bs), :]

    if describe:
        print("========================")
        print(f"PHASE         : {phase}")
        print(f"FILE          : {data_dir / filename}")
        print("========================\n")

        print("===== SUMMARY =====")
        print(f"HUMOR     : {len(dataset[dataset['label'] == 1])}")
        print(f"NON-HUMOR : {len(dataset[dataset['label'] == 0])}")
        print("-------------------")
        print(f"TOTAL     : {dataset.shape[0]}")
        print("===================\n")

    return dataset
