import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from src.comet_commonsense.src.data.utils import TextEncoder
from src.models.keyword_base import KWComHumor
from src.utils import ConstantLRwithWarmup
from transformers import AutoTokenizer


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        *,
        model: nn.Module,
        lr: float,
        epoch: int,
        num_classes: int,
        lr_insert: float = 1e-4,
        lr_scheduler: bool = False,
        warmup_step: int = 0,
        use_keywords: bool = False,
        model_path: str = "bert-base-uncased",
        comet_tokenizer: Optional[TextEncoder] = None,
        comet_learnable: bool = False,
        **kwargs: Any,
    ) -> None:
        super(LightningModel, self).__init__(**kwargs)

        self.use_keywords = use_keywords  # whether to use knowledge base or not

        self.model = model  # model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.comet_tokenizer = comet_tokenizer
        self.comet_learnable = comet_learnable

        # settings for learning model
        self.lr = lr
        self.lr_insert = lr_insert
        self.epoch = epoch
        self.num_classes = num_classes
        self.optimizer = optim.AdamW
        self.use_scheduler = lr_scheduler
        self.warmup_step = warmup_step

        # criterion
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_val = nn.CrossEntropyLoss()

        # metrics
        self.Accuracy = torchmetrics.Accuracy()
        self.Precision = torchmetrics.Precision(multiclass=False)
        self.Recall = torchmetrics.Recall(multiclass=False)
        self.F1 = torchmetrics.F1Score(multiclass=False)
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.ROC = torchmetrics.ROC(pos_label=1)  # multiclass = False?

        self.Accuracy_val = torchmetrics.Accuracy()
        self.Precision_val = torchmetrics.Precision(multiclass=False)
        self.Recall_val = torchmetrics.Recall(multiclass=False)
        self.F1_val = torchmetrics.F1Score(multiclass=False)
        self.confmat_val = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.ROC_val = torchmetrics.ROC(pos_label=1)

        self.cosine_similarity = CosSim()

        if isinstance(self.logger, WandbLogger):
            self.save_hyperparameters()

    def _get_trainable_params(self, model) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        lora_param = 0
        nlora_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if "lora" in name:
                    lora_param += param.numel()
                else:
                    nlora_param += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def forward(self, inputs, keywords=None, *args, **kwargs):
        if self.use_keywords:
            outputs = self.model(inputs, keywords, *args, **kwargs)
            return outputs

        outputs = self.model(inputs, *args, **kwargs)

        return outputs

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        lr_insert = self.lr_insert
        for n, p in self.model.named_parameters():
            if "comet" in n:  # freeze COMET
                p.requires_grad = False
            else:
                p.requires_grad = True

        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (
                        "fusion" in n
                        or "classifier" in n
                        or "pooler" in n
                        or "refinement" in n
                    )
                    and "bias" not in n
                ],
                "lr": lr_insert,
                "name": "lr/insert",
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (
                        "fusion" in n
                        or "classifier" in n
                        or "pooler" in n
                        or "refinement" in n
                    )
                    and "bias" in n
                ],
                "lr": lr_insert,
                "weight_decay": 0.0,
                "name": "lr/insert-nodecay",  # no decay params
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "LayerNorm" not in n
                    and "layer_norm" not in n
                    and "bias" not in n
                    and "comet" not in n
                    and "fusion" not in n
                    and "classifier" not in n
                    and "pooler" not in n
                    and "refinement" not in n
                ],
                "name": "lr/default",
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (
                        "fusion" not in n
                        and "classifier" not in n
                        and "pooler" not in n
                        and "refinement" not in n
                    )
                    and ("LayerNorm" in n or "layer_norm" in n or "bias" in n)
                ],
                "weight_decay": 0.0,
                "name": "lr/default-nodecay",
            },
        ]

        optimizer = self.optimizer(
            params=params,
            lr=self.lr,
        )

        self._get_trainable_params(self.model)

        if self.use_scheduler:  # Warmup
            lr_scheduler = {
                "scheduler": ConstantLRwithWarmup(
                    optimizer=optimizer,
                    warmup_step=self.warmup_step,
                    eta_min=0,
                ),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]

        return [optimizer]

    def _heatmap_dataframe(
        self,
        array: np.ndarray,
        sentences: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(data=array, index=context, columns=sentences)
        return df

    def _make_heatmap(
        self,
        array: np.ndarray,
        sentences: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
        title: Optional[str] = None,
        vmin: Union[float, None] = 0.0,
        vmax: Union[float, None] = 1.0,
        annot: bool = False,
        fmt: str = "d",
        square: bool = False,
        cmap: Optional[str] = None,
        xticklabels: bool = True,
        yticklabels: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        plt.clf()
        data = self._heatmap_dataframe(
            array=array, sentences=sentences, context=context
        )
        heatmap = sns.heatmap(
            data=data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=annot,
            fmt=fmt,
            square=square,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        plt.tight_layout()
        # plt.yticks(rotation=0)
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        heatmap.tick_params(axis="both", which="both", length=0)
        figure = heatmap.get_figure()
        plt.close()
        return figure

    def _save_fig(
        self,
        path: str,
        array: np.ndarray,
        sentences: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
        title: Optional[str] = None,
        vmin: Union[float, None] = 0.0,
        vmax: Union[float, None] = 1.0,
        annot: bool = False,
        fmt: str = "d",
        square: bool = False,
        cmap: Optional[str] = None,
        xticklabels: bool = True,
        yticklabels: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        plt.clf()
        data = self._heatmap_dataframe(
            array=array, sentences=sentences, context=context
        )
        heatmap = sns.heatmap(
            data=data,
            vmin=vmin,
            vmax=vmax,
            annot=annot,
            fmt=fmt,
            square=square,
            cmap=cmap,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        plt.tight_layout()
        # plt.yticks(rotation=0)
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        heatmap.tick_params(axis="both", which="both", length=0)
        plt.savefig(path)
        plt.close()

    def training_step(self, batch, batch_index, *args):
        if self.use_keywords:
            inputs, keywords, labels, idx = batch
            outs = self(
                inputs,
                keywords,
                label=labels,
                indices=idx,
            )
            outputs = outs["outputs"]
            attention_weights = outs["attention_weights"]
        else:
            inputs, labels = batch
            outs = self(inputs)
            outputs = outs["outputs"]

        # training loss
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss.item())

        outputs = torch.argmax(outputs, dim=1)
        acc = self.Accuracy(outputs, labels)
        prec = self.Precision(outputs, labels)
        rec = self.Recall(outputs, labels)
        f1 = self.F1(outputs, labels)
        confmat = self.confmat(outputs, labels)
        roc = self.ROC(outputs, labels)

        if batch_index == 0 and isinstance(self.model, KWComHumor):
            # Attention weights
            attention_weights = [
                attn_weights.detach().cpu().numpy()
                for attn_weights in attention_weights.squeeze(dim=2)
            ]
            sentences = [
                [
                    " ".join(
                        [self.comet_tokenizer.decoder[token.item()] for token in item]
                    )
                    .replace("</w>", " ")
                    .replace("<unk>", "")
                    for item in items
                ]
                for items in keywords["input_ids"]
            ]
            context = [f"head_{i}" for i in range(self.model.num_heads)]
            titles = [
                f"P/L: {pred}/{label}"
                for pred, label in zip(
                    outputs,
                    labels,
                )
            ]
            images = [
                self._make_heatmap(
                    array=attn,
                    sentences=keywords if len(keywords) < 50 else None,
                    context=context,
                    title=title,
                )
                for attn, keywords, title in zip(attention_weights, sentences, titles)
            ]
            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key="train/attention_weights",
                        images=images,
                        caption=[
                            caption.replace("[PAD]", "")
                            .replace("[SEP]", "")
                            .replace("[CLS]", "")
                            .replace("<s>", "")
                            .replace("</s>", "")
                            for caption in self.tokenizer.batch_decode(
                                inputs["input_ids"]
                            )
                        ],
                    )

        return {
            "loss": loss,
            # "preds": outputs,
            # "labels": labels,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": confmat,
            "roc": roc,
        }

    def validation_step(self, batch, batch_index, *args):
        if self.use_keywords:
            inputs, keywords, labels, idx = batch
            outs = self(inputs, keywords, label=labels)
            outputs = outs["outputs"]
            attention_weights = outs["attention_weights"]
        else:
            inputs, labels = batch
            outs = self(inputs)
            outputs = outs["outputs"]

        loss = self.criterion(outputs, labels)
        self.log("valid/loss", loss.item())

        probs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        acc = self.Accuracy_val(outputs, labels)
        prec = self.Precision_val(outputs, labels)
        rec = self.Recall_val(outputs, labels)
        f1 = self.F1_val(outputs, labels)
        confmat = self.confmat_val(outputs, labels)
        roc = self.ROC_val(outputs, labels)

        if (
            batch_index == 0
            and isinstance(self.model, KWComHumor)
            and self.comet_tokenizer is not None
        ):
            attention_weights = [
                attn_weights.detach().cpu().numpy()
                for attn_weights in attention_weights.squeeze(dim=2)
            ]
            sentences = [
                [
                    " ".join(
                        [self.comet_tokenizer.decoder[token.item()] for token in item]
                    )
                    .replace("</w>", " ")
                    .replace("<unk>", "")
                    for item in items
                ]
                for items in keywords["input_ids"]
            ]
            context = [f"head_{i}" for i in range(self.model.num_heads)]
            titles = [
                f"P/L: {pred}/{label} ({prob[pred] * 100:.3f}%)"
                for pred, label, prob in zip(outputs, labels, probs)
            ]
            images = [
                self._make_heatmap(
                    array=attn,
                    sentences=keywords,
                    context=context,
                    title=title,
                    xticklabels=False if len(keywords) > 50 else True,
                )
                for attn, keywords, title in zip(attention_weights, sentences, titles)
            ]

            for logger in self.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(
                        key="valid/attention_weights",
                        images=images,
                        caption=[
                            caption.replace("[PAD]", "")
                            .replace("[SEP]", "")
                            .replace("[CLS]", "")
                            .replace("<s>", "")
                            .replace("</s>", "")
                            for caption in self.tokenizer.batch_decode(
                                inputs["input_ids"]
                            )
                        ],
                    )
        return {
            "loss": loss,
            # "preds": outputs,
            # "probs": probs,
            # "labels": labels,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": confmat,
            "roc": roc,
        }

    def test_step(self, batch, batch_index, *args):
        if self.use_keywords:
            inputs, keywords, labels, idx = batch
            outs = self(inputs, keywords, label=labels)
            outputs = outs["outputs"]
            attention_weights = outs["attention_weights"]
        else:
            inputs, labels = batch
            outs = self(inputs)
            outputs = outs["outputs"]

        loss = self.criterion(outputs, labels)

        outputs = torch.argmax(outputs, dim=1)
        acc = self.Accuracy(outputs, labels)
        prec = self.Precision(outputs, labels)
        rec = self.Recall(outputs, labels)
        f1 = self.F1(outputs, labels)
        confmat = self.confmat(outputs, labels)
        roc = self.ROC(outputs, labels)

        if self.use_keywords and batch_index < 5:
            os.makedirs(
                os.path.join(self.trainer.log_dir, "images", "attention_weights"),
                exist_ok=True,
            )
            attention_weights = [
                attn_weights.detach().cpu().numpy()
                for attn_weights in attention_weights.squeeze(dim=2)
            ]
            sentences = [
                [
                    " ".join(
                        [self.comet_tokenizer.decoder[token.item()] for token in item]
                    )
                    .replace("</w>", " ")
                    .replace("<unk>", "")
                    for item in items
                ]
                for items in keywords["input_ids"]
            ]
            context = [f"head_{i}" for i in range(self.model.num_heads)]
            titles = [
                f"P/L: {pred}/{label}"
                for pred, label in zip(
                    outputs,
                    labels,
                )
            ]
            save_to = os.path.join(self.trainer.log_dir, "images", "attention_weights")
            for idx, (attn, keywords, title) in enumerate(
                zip(attention_weights, sentences, titles)
            ):
                self._save_fig(
                    os.path.join(save_to, f"image_{len(batch) * batch_index + idx}"),
                    array=attn,
                    sentences=keywords,
                    context=context,
                    title=title,
                    xticklabels=False if len(keywords) > 50 else True,
                )

        return {
            "loss": loss,
            "preds": outputs,
            "labels": labels,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": confmat,
            "roc": roc,
        }

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        if self.use_keywords:
            inputs, keywords, *_ = batch
            outs = self(inputs, keywords)
            outputs = outs["outputs"]
        else:
            inputs, *_ = batch
            outs = self(inputs)

        outputs = outs["outputs"]

        probs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)

        return outputs, probs

    def training_epoch_end(self, outputs):
        # loggnig when using torchmetrics
        self.log("train/accuracy", self.Accuracy.compute())
        self.log("train/precision", self.Precision.compute())
        self.log("train/recall", self.Recall.compute())
        self.log("train/f1", self.F1.compute())
        confmat = self.confmat.compute()
        self.confmat.reset()
        fpr, tpr, threshold = self.ROC.compute()
        self.ROC.reset()

        image = self._make_heatmap(
            array=confmat.detach().cpu().numpy(),
            sentences=["Not Humorous", "Humorous"],
            context=["Not Humorous", "Humorous"],
            vmin=None,
            vmax=None,
            annot=True,
            square=True,
            xlabel="Predict",
            ylabel="Ground Truth",
            cmap="Blues",
        )
        plt.clf()
        fig = plt.figure()
        plt.plot(fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy())
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image(key="train/confusion_matrix", images=[image])
                logger.log_image(key="train/roc", images=[fig])

        # logging when not using torchmetrics
        loss = 0.0
        i = 0
        for batch in outputs:
            if isinstance(batch, list):
                for mini in batch:
                    i += 1
                    loss += mini["loss"].detach()
            else:
                i += 1
                loss += batch["loss"].detach()

        self.log("train/loss", loss / i)
        wandb.log(data={}, commit=True)

    def validation_epoch_end(self, outputs):
        # loggnig when using torchmetrics
        self.log("valid/accuracy", self.Accuracy_val.compute())
        self.log("valid/precision", self.Precision_val.compute())
        self.log("valid/recall", self.Recall_val.compute())
        self.log("valid/f1", self.F1_val.compute())
        confmat = self.confmat_val.compute()
        self.confmat_val.reset()
        fpr, tpr, threshold = self.ROC_val.compute()
        self.ROC_val.reset()
        image = self._make_heatmap(
            array=confmat.detach().cpu().numpy(),
            sentences=["Not Humorous", "Humorous"],
            context=["Not Humorous", "Humorous"],
            vmin=None,
            vmax=None,
            annot=True,
            square=True,
            xlabel="Predict",
            ylabel="Ground Truth",
            cmap="Blues",
        )
        plt.clf()
        fig = plt.figure()
        plt.plot(fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy())
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image(key="valid/confusion_matrix", images=[image])
                logger.log_image(key="valid/roc", images=[fig])

        # logging when not using torchmetrics
        loss = 0.0
        i = 0
        for batch in outputs:
            if isinstance(batch, list):
                for mini in batch:
                    i += 1
                    loss += mini["loss"]
            else:
                i += 1
                loss += batch["loss"]

        self.log("valid/loss", loss / i)

    def test_epoch_end(self, outputs):
        # loggnig when using torchmetrics
        self.log("test/accuracy", self.Accuracy.compute())
        self.log("test/precision", self.Precision.compute())
        self.log("test/recall", self.Recall.compute())
        self.log("test/f1", self.F1.compute())
        os.makedirs(os.path.join(self.trainer.log_dir, "images"), exist_ok=True)
        confmat = self.confmat.compute()
        fpr, tpr, threshold = self.ROC.compute()

        self.confmat.reset()
        self._save_fig(
            path=os.path.join(self.trainer.log_dir, "images", "confusion_matrix"),
            array=confmat.detach().cpu().numpy(),
            sentences=["Not Humorous", "Humorous"],
            context=["Not Humorous", "Humorous"],
            vmin=None,
            vmax=None,
            annot=True,
            square=True,
            xlabel="Predict",
            ylabel="Ground Truth",
            cmap="Blues",
        )
        plt.clf()
        fig = plt.figure()
        plt.plot(fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy())
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(os.path.join(self.trainer.log_dir, "images", "roc_curve"))
        fig.clear()

        # logging when not using torchmetrics
        loss = 0.0
        i = 0
        for batch in outputs:
            if isinstance(batch, list):
                for mini in batch:
                    i += 1
                    loss += mini["loss"]
            else:
                i += 1
                loss += batch["loss"]

        self.log("test/loss", loss / i)


class CosSim(object):
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        cossim = self._make_cossim_matrix(tensor)
        return cossim

    def _make_cossim_matrix(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.detach().cpu().numpy()
        cossim = np.array(
            [
                np.dot(batch, batch.T)
                / (
                    np.linalg.norm(batch, axis=1, keepdims=True)
                    * np.linalg.norm(batch.T, axis=0, keepdims=True)
                )
                for batch in tensor
            ]
        )
        return cossim
