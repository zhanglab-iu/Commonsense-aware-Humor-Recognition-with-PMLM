import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import create_comet, load_dataset

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


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super(PositionalEncoding, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs: torch.Tensor):
        if inputs.ndim < 3:
            inputs = inputs.unsqueeze(dim=-1)
        num_positions = inputs.size(-2)
        position_embeddings = torch.as_tensor(
            range(self.vocab_size, self.vocab_size + num_positions),
            dtype=torch.long,
            device=inputs.device,
        )
        inputs = inputs.repeat(1, 1, 2)
        inputs[:, :, 1] = position_embeddings
        return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--task", type=str, required=True, help="HaHackathon | Humicroedit"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(".cache/")
    output_dir.mkdir(exist_ok=True, parents=True)

    comet_model, comet_tokenizer = create_comet(
        "pretrained/conceptnet_pretrained_model.pickle"
    )
    comet = comet_model.transformer.to(args.device)
    pe = PositionalEncoding(40514).to(args.device)

    traindf = load_dataset(
        data_dir=input_dir,
        task=args.task,
        phase="train",
    )

    # 5 relations
    relations = [
        comet_tokenizer.encode([rel], verbose=True)[0]
        for rel in ("Causes", "IsA", "HasA", "Desires", "UsedFor")
    ]
    max_length_comet = 10 + max((len(rel) for rel in relations))
    five_path = output_dir / args.task / "5rels"
    five_path.mkdir(exist_ok=True, parents=True)
    for idx, candidates in enumerate(tqdm(traindf.keywords)):
        keywords_sets = comet_tokenizer.encode(candidates[:6], verbose=False)
        keywords = (
            np.stack(
                [
                    np.array(
                        keyword
                        + rel
                        + [0] * (max_length_comet - len(keyword) - len(rel)),
                        dtype=np.int64,
                    )
                    if len(keyword) + len(rel) <= max_length_comet
                    else np.array(
                        keyword[: max_length_comet - len(rel)] + rel,
                        dtype=np.int64,
                    )
                    for keyword in keywords_sets
                    for rel in relations
                ],
                axis=0,
            )
            if len(keywords_sets) > 0
            else np.array([[0] * max_length_comet], dtype=np.int64)
        )

        input_ids = torch.from_numpy(keywords).to(args.device)

        with torch.no_grad():
            mask = (input_ids != 0).float().to(args.device)
            outputs = pe(input_ids)
            outputs = comet(outputs.unsqueeze(dim=1), sequence_mask=mask)
        for i, ids in enumerate(input_ids):
            if not (ids == 0).all():
                outputs[i][-1] = outputs[i][torch.nonzero(ids).max().item()]
        outputs = outputs[:, -1, :]
        torch.save(outputs.cpu(), five_path / f"{idx}.pt")

    # All relations
    relations = [comet_tokenizer.encode([rel], verbose=True)[0] for rel in RELATIONS]
    max_length_comet = 10 + max((len(rel) for rel in relations))
    all_path = output_dir / args.task / "All"
    all_path.mkdir(exist_ok=True, parents=True)
    for idx, candidates in enumerate(tqdm(traindf.keywords)):
        keywords_sets = comet_tokenizer.encode(candidates[:6], verbose=False)
        keywords = (
            np.stack(
                [
                    np.array(
                        keyword
                        + rel
                        + [0] * (max_length_comet - len(keyword) - len(rel)),
                        dtype=np.int64,
                    )
                    if len(keyword) + len(rel) <= max_length_comet
                    else np.array(
                        keyword[: max_length_comet - len(rel)] + rel,
                        dtype=np.int64,
                    )
                    for keyword in keywords_sets
                    for rel in relations
                ],
                axis=0,
            )
            if len(keywords_sets) > 0
            else np.array([[0] * max_length_comet], dtype=np.int64)
        )

        input_ids = torch.from_numpy(keywords).to(args.device)

        with torch.no_grad():
            mask = (input_ids != 0).float().to(args.device)
            outputs = pe(input_ids)
            outputs = comet(outputs.unsqueeze(dim=1), sequence_mask=mask)
        for i, ids in enumerate(input_ids):
            if not (ids == 0).all():
                outputs[i][-1] = outputs[i][torch.nonzero(ids).max().item()]
        outputs = outputs[:, -1, :]
        torch.save(outputs.cpu(), all_path / f"{idx}.pt")
