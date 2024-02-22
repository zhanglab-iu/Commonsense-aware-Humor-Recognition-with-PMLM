from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class LMBaseModel(nn.Module):
    def __init__(
        self,
        *,
        model_path: str,
        pretrained: bool = True,
        model_type: Union[str, None] = None,
        num_label: int = 2,
        dropout: Optional[float] = None,
        **kwargs,
    ) -> None:
        assert pretrained is False or model_type is None
        super(LMBaseModel, self).__init__()
        if pretrained:
            config = AutoConfig.from_pretrained(model_path)
        else:
            if isinstance(model_type, str):
                config = AutoConfig.for_model(model_type=model_type)
            else:
                raise ValueError
        for key, value in kwargs.items():
            if getattr(config, key):
                setattr(config, key, value)

        hidden_size = config.hidden_size
        if dropout is None:
            dropout = config.hidden_dropout_prob

        dropout_args = {}
        if "distilbert" in model_path:
            dropout_args["dropout"] = dropout
        else:
            dropout_args["hidden_dropout_prob"] = dropout
        self.backbone = AutoModel.from_pretrained(model_path, **dropout_args)
        if "large" in model_path:
            self.backbone.gradient_checkpointing_enable()

        self.pooler = Pooler(input_size=hidden_size)

        self.classifier = nn.Linear(hidden_size, num_label)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.0)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cls_repr = self.backbone(**inputs)["last_hidden_state"][:, 0, :]
        pooled_outputs = self.pooler(cls_repr)
        outputs = self.classifier(pooled_outputs)
        return {
            "outputs": outputs,
        }


class Pooler(nn.Module):
    """ClassifierHead
    This is used for classification when fine-tuning
    dense -> tanh
    """

    def __init__(self, input_size: int) -> None:
        super(Pooler, self).__init__()
        self.dense = nn.Linear(in_features=input_size, out_features=input_size)
        self.activation = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.dense.weight)
        self.dense.bias.data.fill_(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        outputs = self.activation(outputs)

        return outputs


class MLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.dense.weight)
        self.dense.bias.data.fill_(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dense(inputs)
        outputs = self.activation(outputs)

        return outputs


class Similarity(nn.Module):
    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim = self.cos(x, y)
        return sim / self.temp
