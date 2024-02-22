import os
import sys
from collections import defaultdict
from typing import List, Optional, Tuple, Union

from src.comet_commonsense.src.data.utils import TextEncoder

if sys.version_info > (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Define relations for COMET
RELATIONS = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubevent",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
]


# Dataset
class HumorDataset(Dataset):
    """HumorDataset"""

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        label: Union[str, int] = "label",
    ) -> None:
        super().__init__()
        self.df = df
        self.label = label
        self.texts = df["text"].to_numpy()
        self.labels = df[label].to_numpy().astype(np.int32)
        self.keywords = df["keywords"].to_numpy()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Union[Tuple[str, int], Tuple[str, str, int]]:
        text = self.texts[idx]
        label = self.labels[idx]
        keywords = self.keywords[idx]
        return text, keywords, label, idx


class CollateFn:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        num_classes: int = 2,
        use_keywords: bool = False,
        comet_tokenizer: Optional[TextEncoder] = None,
        input_method: Optional[Literal["head", "tail", "head-tail"]] = None,
        head_ratio: float = 0.5,
        relation: Union[str, List[str]] = "Causes",
        max_keyword_legth: int = 3,
        max_keyword_num: int = 6,
    ) -> None:
        if use_keywords and comet_tokenizer is None:
            raise ValueError
        if isinstance(relation, str):  # relation as list
            if relation.lower() == "all":
                relation = RELATIONS
            else:
                relation = [relation]
        elif isinstance(relation, list):
            if "all" in [rel.lower() for rel in relation]:
                relation = RELATIONS

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        self.use_keywords = use_keywords
        self.comet_tokenizer = comet_tokenizer
        self.relation: list = relation
        self.input_method = input_method
        self.max_keyword_length = max_keyword_legth
        self.max_keyword_num = max_keyword_num

        self.head_ratio = head_ratio  # used for head-tail input
        self.head_size = int(max_length * head_ratio)
        self.tail_size = max_length - self.head_size
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # prepare relation token
        if self.comet_tokenizer is not None:
            self.relation = [
                self.comet_tokenizer.encode([rel], verbose=False)[0]
                for rel in self.relation
            ]
            self.max_length_comet = 10 + max((len(rel) for rel in self.relation))

    def __call__(self, data):
        texts, keywords, labels, indices = zip(*data)
        texts = list(texts)
        keywords = list(keywords)
        labels = list(labels)
        indices = list(indices)

        if self.input_method in (None, "head"):
            texts = self.tokenizer.batch_encode_plus(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        elif self.input_method in ("tail", "head-tail"):
            texts = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                truncation=False,
                return_tensors="np",
            )
            new_texts = defaultdict(list)
            for batch in zip(*texts.values()):
                if len(batch) == 3:
                    input_ids, _, attention_mask = batch
                elif len(batch) == 2:
                    input_ids, attention_mask = batch
                input_ids = input_ids[attention_mask == 1]
                if input_ids.shape[0] >= self.max_length:
                    if self.input_method == "tail":
                        input_ids = input_ids[-self.max_length :]
                        input_ids[0] = self.tokenizer.cls_token_id
                    else:  # self.input_method == "head-tail"
                        new_input_ids = np.zeros(self.max_length, dtype=np.int64)
                        new_input_ids[: self.head_size] = input_ids[: self.head_size]
                        new_input_ids[-self.tail_size :] = input_ids[-self.tail_size :]
                        input_ids = new_input_ids
                    attention_mask = np.ones(self.max_length, dtype=np.int64)
                else:
                    length = input_ids.shape[0]
                    input_ids = np.concatenate(
                        [
                            input_ids,
                            np.zeros(self.max_length - length, dtype=np.int64),
                        ]
                    )
                    attention_mask = np.zeros(self.max_length, dtype=np.int64)
                    attention_mask[:length] = np.ones(length, dtype=np.int64)
                new_texts["input_ids"].append(input_ids)
                new_texts["attention_mask"].append(attention_mask)

            texts = {
                k: torch.from_numpy(np.stack(v, axis=0)) for k, v in new_texts.items()
            }

        if self.use_keywords:  # if using keywords
            bs = len(keywords)  # keywords
            keywords_sets = [
                self.comet_tokenizer.encode(
                    batch[: self.max_keyword_num], verbose=False
                )
                for batch in keywords
            ]  # (batch, keywords, length <= 3)
            keywords = np.zeros(
                (bs, self.max_keyword_num * len(self.relation), self.max_length_comet),
                dtype=np.int64,
            )
            keywords_sets = [
                (
                    np.stack(
                        [
                            (
                                np.array(
                                    keyword
                                    + rel
                                    + [0]
                                    * (self.max_length_comet - len(keyword) - len(rel)),
                                    dtype=np.int64,
                                )
                                if len(keyword) + len(rel) <= self.max_length_comet
                                else np.array(
                                    keyword[: self.max_length_comet - len(rel)] + rel,
                                    dtype=np.int64,
                                )
                            )
                            for keyword in batch
                            for rel in self.relation
                        ],
                        axis=0,
                    )
                    if len(batch) > 0  # if any keyword exists
                    else np.array([[0] * self.max_length_comet], dtype=np.int64)
                )
                for batch in keywords_sets
            ]
            for idx, keyword in enumerate(keywords_sets):
                keywords[idx, : keyword.shape[0], :] = keyword
            # assert keywords.size(0) == bs
            # assert keywords.size(1) == self.max_keyword_num
            # assert keywords.size(2) == self.max_length_comet

            attention_mask = np.where(
                (keywords == 0).all(axis=2), 1, 0
            )  # 1: mask 0: not mask
            # assert attention_mask.size(0) == bs, attention_mask.size(0)
            # assert attention_mask.size(1) == self.max_keyword_num, attention_mask.size(
            #     2
            # )

            keywords = {
                "input_ids": torch.from_numpy(keywords),
                "attention_mask": torch.from_numpy(attention_mask),
            }
            labels = torch.as_tensor(labels, dtype=torch.long)  # labels to torch tensor
            return (
                texts,
                keywords,
                labels,
                indices,
            )
        labels = torch.as_tensor(labels, dtype=torch.long)  # labels to torch tensor
        return texts, labels  # if not using knowledge


class LightningDataRetriever(pl.LightningDataModule):
    def __init__(
        self,
        *,
        traindf: pd.DataFrame,
        model_path: str,
        validdf: Union[pd.DataFrame, None] = None,
        testdf: Union[pd.DataFrame, None] = None,
        label: Union[str, int] = "label",
        # max_sample: Optional[int] = None,
        use_keywords: bool = False,
        comet_tokenizer: Optional[TextEncoder] = None,
        relation: Union[str, List[str]] = "Causes",
        max_keyword_num: int = 6,
        seed: int = 42,
        batch_size: int = 16,
        max_length: int = 128,
        input_method: Optional[Literal["head", "tail", "head-tail"]] = None,
        head_ratio: float = 0.5,
    ):
        """
        Args
        ----
        config: Dict[str, Any]
            Experiment Configuration
        df    : pd.DataFrame
            DataFrame includes humor data
        """
        super(LightningDataRetriever, self).__init__()
        self.seed = seed
        self.batch_size = batch_size
        # self.max_sample = max_sample
        self.stratified = True
        self.use_keywords = use_keywords
        self.comet_tokenizer = comet_tokenizer
        self.relation = relation
        self.traindf = traindf
        self.validdf = validdf
        self.testdf = testdf
        self.cols: np.ndarray = traindf.columns.to_numpy()
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        self.max_length = max_length
        self.input_method = input_method
        self.head_ratio = head_ratio
        self.max_keyword_num = max_keyword_num

    def prepare_data(self) -> None:
        """
        Preparing Dataset by spliting DataFrame for training into training set and validation set
        when validation DataFrame does not exist
        """
        return

    def train_dataloader(self) -> DataLoader:
        traindf = self.traindf
        print("Setup training set")
        print("Input method :", self.input_method)
        trainset = HumorDataset(df=traindf, label=self.label)

        # Repproducibility
        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            dataset=trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=CollateFn(
                self.tokenizer,
                self.max_length,
                use_keywords=self.use_keywords,
                comet_tokenizer=self.comet_tokenizer,
                relation=self.relation,
                input_method=self.input_method,
                head_ratio=self.head_ratio,
                max_keyword_num=self.max_keyword_num,
            ),
            generator=g,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        validdf = self.validdf
        print("Setup validation set")
        validset = HumorDataset(df=validdf, label=self.label)
        return DataLoader(
            dataset=validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=CollateFn(
                self.tokenizer,
                self.max_length,
                use_keywords=self.use_keywords,
                comet_tokenizer=self.comet_tokenizer,
                relation=self.relation,
                input_method=self.input_method,
                head_ratio=self.head_ratio,
                max_keyword_num=self.max_keyword_num,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        if self.testdf is None:
            raise AttributeError("This LightningDataModule does not have `testdf`")
        testdf = self.testdf
        print("Setup test set")
        testset = HumorDataset(df=testdf, label=self.label)
        return DataLoader(
            dataset=testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=CollateFn(
                self.tokenizer,
                self.max_length,
                use_keywords=self.use_keywords,
                comet_tokenizer=self.comet_tokenizer,
                relation=self.relation,
                input_method=self.input_method,
                head_ratio=self.head_ratio,
                max_keyword_num=self.max_keyword_num,
            ),
        )

    def predict_dataloader(self) -> DataLoader:
        if self.testdf is None:
            raise AttributeError("This LightningDataModule does not have `testdf`")
        testdf = self.testdf
        testset = HumorDataset(df=testdf, label=self.label)
        return DataLoader(
            dataset=testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=CollateFn(
                self.tokenizer,
                self.max_length,
                use_keywords=self.use_keywords,
                comet_tokenizer=self.comet_tokenizer,
                relation=self.relation,
                input_method=self.input_method,
                head_ratio=self.head_ratio,
                max_keyword_num=self.max_keyword_num,
            ),
        )
