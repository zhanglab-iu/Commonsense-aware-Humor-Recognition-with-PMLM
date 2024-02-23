import json
import warnings
from argparse import ArgumentParser
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import yake
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import src.comet_commonsense.src.interactive.functions as interactive
from src.comet_commonsense.src.data.utils import TextEncoder
from src.comet_commonsense.src.evaluate.sampler import Sampler
from src.comet_commonsense.src.models.gpt import LMHead, LMModel, TransformerModel
from src.models import KWComHumor, LightningModel
from src.utils import CollateFn, HumorDataset, load_all

warnings.simplefilter("ignore")
sns.set_theme()


class COMET(nn.Module):
    """Transformer with language model head only"""

    def __init__(
        self,
        transformer: TransformerModel,
        lm_head: LMHead,
        vocab: int = 40990,
        n_ctx: int = 31,
        return_probs: bool = False,
        return_acts: bool = False,
    ):
        super(COMET, self).__init__()
        self.transformer = transformer
        self.lm_head = lm_head
        self.return_probs = return_probs
        self.return_acts = return_acts
        if self.return_probs or self.return_acts:
            pos_emb_mask = torch.zeros(1, 1, vocab)
            pos_emb_mask[:, :, -n_ctx:] = -1e12
            self.register_buffer("pos_emb_mask", pos_emb_mask)

    def forward(self, x, sequence_mask=None):
        h = self.transformer(x, sequence_mask)
        lm_logits = self.lm_head(h)
        if self.return_probs:
            lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        elif self.return_acts:
            lm_logits = lm_logits + self.pos_emb_mask
        return lm_logits


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    return parser


def load_config(filename: str) -> dict:
    config = json.load(open(filename, "r"), parse_int=int, parse_float=float)
    return config


def load_comet(
    ckpt: str,
    config: dict,
    comet_model: LMModel,
) -> Tuple[KWComHumor, TextEncoder]:
    # model setup
    model_path = config.get("model_path", "")
    pretrained = config.get("pretrained", True)
    model_type = config.get("model_type")
    num_classes = config.get("num_classes", 2)
    vocab_size = len(comet_tokenizer.encoder)
    backbone = KWComHumor(
        model_path_lm=model_path,
        comet_model=comet_model,
        pretrained_lm=pretrained,
        model_type_lm=model_type,
        vocab_size=vocab_size,
        num_label=num_classes,
    )
    # lightning setup
    pl_model = LightningModel.load_from_checkpoint(
        ckpt,
        model=backbone,
        lr=0.0,
        epoch=0,
        num_classes=num_classes,
        use_keywords=True,
    )
    # retrieve model
    model = pl_model.model
    return model, comet_tokenizer


def load_tokenizer(config: dict) -> AutoTokenizer:
    model_path = config.get("model_path", "")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def load_lm(
    transformer: TransformerModel,
    lm_head: LMHead,
    tokenizer: TextEncoder,
    data_loader,
):
    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    model = COMET(
        transformer=transformer,
        lm_head=lm_head,
        vocab=len(tokenizer.encoder) + n_ctx,
        n_ctx=n_ctx,
        return_acts=True,
    )
    return model


def load_dataloader(
    config: dict, dataset: HumorDataset, comet_tokenizer: TextEncoder
) -> DataLoader:
    tokenizer = load_tokenizer(config=config)
    max_length = config.get("max_length", 128)
    num_classes = config.get("num_classes", 2)
    input_method = config.get("input_method", "head-tail")
    head_ratio = config.get("head_ratio", 0.75)
    relation = config.get("relation", "Causes")
    collate_fn = CollateFn(
        tokenizer=tokenizer,
        max_length=max_length,
        num_classes=num_classes,
        use_keywords=True,
        comet_tokenizer=comet_tokenizer,
        input_method=input_method,
        head_ratio=head_ratio,
        relation=relation,
        max_keyword_legth=3,
        max_keyword_num=20,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return dataloader


def extract_keywords(extractor: yake.KeywordExtractor, text: str) -> list:
    keywords = extractor.extract_keywords(text=text)
    keywords = [keyword[0] for keyword in keywords]
    return keywords


def generate(
    model: COMET,
    keywords: List[str],
    sampler: Sampler,
    data_loader,
    text_encoder: TextEncoder,
    relation: str,
):
    outputs = [
        interactive.get_conceptnet_sequence(
            e1=keyword,
            model=model,
            sampler=sampler,
            data_loader=data_loader,
            text_encoder=text_encoder,
            relation=relation,
        )
        for keyword in keywords
    ]
    return outputs


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    config = load_config(args.config)
    model, comet_tokenizer, opt, data_loader = load_all(
        model_path=config.get("comet_path", "")
    )
    sampler = interactive.set_sampler(
        opt=opt, sampling_algorithm="beam-3", data_loader=data_loader
    )
    model, comet_tokenizer = load_comet(
        ckpt=args.ckpt, config=config, comet_model=model
    )
    tokenizer = load_tokenizer(config=config)
    lm_model = load_lm(
        transformer=model.comet,
        lm_head=model.lm_head,
        tokenizer=comet_tokenizer,
        data_loader=data_loader,
    )

    extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.1,
        dedupFunc="seqm",
        windowsSize=1,
        top=20,
    )

    while True:
        text = input("input text: ")
        # keywords
        print("Fetching keywords...")
        keywords = extract_keywords(extractor=extractor, text=text)
        print(keywords, "are extracted from", f"`{text}`")
        raw_keywords = keywords.copy()

        df = pd.DataFrame({"text": [text], "label": [np.nan], "keywords": [keywords]})
        dataset = HumorDataset(df=df)
        dataloader = load_dataloader(
            config=config, dataset=dataset, comet_tokenizer=comet_tokenizer
        )

        print("Infering...")
        model.eval()
        with torch.no_grad():
            for texts, keywords, _ in dataloader:
                break
            _ = model(texts, keywords)
            attention_weights = model.attention_weights
            # comet_embeddings = model.comet_embeddings
        print("done!")

        # knowledges
        relation = config.get("relation", "Causes")
        print("Generating knowledges from keywords")
        knowledges = generate(
            model=lm_model,
            keywords=raw_keywords,
            sampler=sampler,
            data_loader=data_loader,
            text_encoder=comet_tokenizer,
            relation=relation,
        )
        print(knowledges)
        print("done!")

        # reform knowledges
        knowledges = [
            knowledge[relation]["e1"]
            + " "
            + relation
            + " : "
            + "\n".join(knowledge[relation]["beams"])
            for knowledge in knowledges
        ]
        display_text = " ".join(
            [
                word + "\n" if i % 20 == 0 else word
                for i, word in enumerate(text.split(), 1)
            ]
        )
        keywords_num = attention_weights.size(-1)
        df_attn = pd.DataFrame(
            data=attention_weights[0].numpy(),
            columns=[knowledges + ["None"] * (keywords_num - len(raw_keywords))],
        )
        plt.figure(figsize=(14, 6))
        ax = sns.heatmap(data=df_attn, vmin=0, vmax=1, square=False, cbar=False)
        plt.tight_layout()
        plt.title(display_text)
        plt.yticks(rotation=90)
        save = input("if you want to save this image, please type target filename: ")
        if save == "":
            plt.show()
            continue
        else:
            plt.savefig(save)
            plt.show()
