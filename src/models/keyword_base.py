from pathlib import Path
from typing import Dict, List, Optional

import einops
import torch
import torch.nn as nn
from src.comet_commonsense.src.models.gpt import LMModel
from transformers import AutoModel


class KWComHumor(nn.Module):
    def __init__(
        self,
        *,
        model_path_lm: str,
        comet_model: LMModel,
        model_type_lm: Optional[str] = None,
        vocab_size: int = 40514,
        num_label: int = 2,
        num_heads: int = 8,
        embedding_dim: int = 768,  # Transformer hidden size
        dropout: float = 0.1,
        fusion_dropout_prob: float = 0.0,
        keywords_dropout_prob: float = 0.0,
        norm: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(KWComHumor, self).__init__()

        # PMLM Setup
        if model_type_lm is None:
            model_type_lm = model_path_lm

        dropout_args = {}
        if "distilbert" in model_path_lm:
            dropout_args["dropout"] = dropout
        else:
            dropout_args["hidden_dropout_prob"] = dropout
        self.backbone_lm = AutoModel.from_pretrained(model_path_lm, **dropout_args)
        if "large" in model_path_lm:
            self.backbone_lm.gradient_checkpointing_enable()

        self.pe = PositionalEncoding(vocab_size=vocab_size)

        # COMET Setup
        self.comet = comet_model.transformer
        self.lm_head = comet_model.lm_head

        hidden_size_lm = self.backbone_lm.config.hidden_size

        # Commonsense-Aware Multi-Head Dot-Product Attention
        self.num_heads = num_heads
        self.fusion = MultiHeadDotProductAttention(
            num_heads=num_heads,
            trainable=True,
            query_size=hidden_size_lm,
            context_size=embedding_dim,
            dropout=fusion_dropout_prob,
            attention_dropout=keywords_dropout_prob,
            return_attention_weights=True,
            bias=False,
            norm=norm,
        )

        # Final Linear
        self.classifier = nn.Linear(hidden_size_lm, num_label)

        # context refinement
        self.context_refinement = nn.ModuleDict(
            {
                "dense0": nn.Linear(hidden_size_lm, hidden_size_lm),
                "activation": nn.Tanh(),  # tanh
                "dense1": nn.Linear(hidden_size_lm, hidden_size_lm),
            }
        )
        # commonsense refinement
        self.commonsense_refinement = nn.ModuleDict(
            {
                "dense0": nn.Linear(hidden_size_lm, hidden_size_lm),
                "activation": nn.Tanh(),  # tanh
                "dense1": nn.Linear(hidden_size_lm, hidden_size_lm),
            }
        )

        self.cache_dir = cache_dir

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.0)

    def make_attention_mask(sequences: torch.Tensor):
        return (sequences != 0).float().to(sequences.device)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        keywords: Dict[str, torch.Tensor],
        indices: Optional[List[int]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Context Encoder
        outputs_lm = self.backbone_lm(**inputs)
        cls_repr = outputs_lm["last_hidden_state"][:, 0, :]  # CLS representation

        # Context Refinement
        refined_context = self.context_refinement["dense0"](cls_repr)
        refined_context = self.context_refinement["activation"](refined_context)
        refined_context = self.context_refinement["dense1"](refined_context)

        # COMET part in Commonsense Acquisition Module
        outputs_keywords_sets = []
        if (
            self.cache_dir is not None and indices is not None
        ):  # load commonsense embeddings from cache
            outputs_keywords = torch.stack(
                [torch.load(Path(self.cache_dir) / f"{index}.pt") for index in indices],
                dim=0,
            ).to(cls_repr.device)
        else:
            with torch.no_grad():
                for batch in keywords["input_ids"]:
                    # token-level attention mask
                    sequence_mask = (
                        (batch != 0).float().to(batch.device)
                    )  # maybe 0 means mask?? yes!!
                    # Positional Encoding
                    batch = self.pe(batch)
                    outputs_keywords_sets.append(
                        self.comet(batch.unsqueeze(dim=1), sequence_mask=sequence_mask)
                    )

                attention_mask = keywords.get(
                    "attention_mask"
                )  # sentence-level attention mask
                outputs_keywords = torch.stack(
                    outputs_keywords_sets, dim=0
                )  # (batch, keywords, tokens, embedding_dim)

                for i, input_ids in enumerate(keywords["input_ids"]):
                    for j, ids in enumerate(input_ids):
                        if not (ids == 0).all():
                            outputs_keywords[i][j][-1] = outputs_keywords[i][j][
                                torch.nonzero(ids).max().item()
                            ]
                outputs_keywords = outputs_keywords[
                    :, :, -1, :
                ]  # use only last　token hidden state -> (batch, keywords, embedding_dim)

        # Commonsense-aware Multi-Head Attention (CA-MHA)
        commonsense, attention_weights = self.fusion(
            cls_repr.unsqueeze(dim=1),
            outputs_keywords,
            attention_mask=attention_mask,
        )
        commonsense = commonsense.squeeze(dim=1)  # squash dim on dim1

        # Commonsense Refinement
        refined_commonsense = self.commonsense_refinement["dense0"](commonsense)
        refined_commonsense = self.commonsense_refinement["activation"](
            refined_commonsense
        )
        refined_commonsense = self.commonsense_refinement["dense1"](refined_commonsense)

        # Infusion
        fused_outputs = refined_context + refined_commonsense

        # Classifier
        outputs = self.classifier(fused_outputs)

        return {
            "outputs": outputs,
            "attention_weights": attention_weights,
        }


class PositionalEncoding(nn.Module):
    """Positional Encoding"""

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


class MultiHeadDotProductAttention(nn.Module):
    """
    Simple MultiHeadDotProductAttention
    Query, Key, Value -> Text, Knowledge, Knowledge
    """

    def __init__(
        self,
        num_heads: int,
        trainable: bool = True,
        query_size: int = 768,
        context_size: int = 768,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        return_attention_weights: bool = True,
        bias: bool = False,
        norm: bool = False,
    ) -> None:
        super(MultiHeadDotProductAttention, self).__init__()
        assert query_size % num_heads == 0 and context_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = query_size // num_heads
        self.trainable = trainable
        self.return_attention_weights = return_attention_weights
        self.register_buffer(name="attention_weights", tensor=None, persistent=False)
        if trainable:
            self.QW = nn.Linear(query_size, context_size, bias=bias)
            self.KW = nn.Linear(context_size, context_size, bias=bias)
            self.VW = nn.Linear(context_size, context_size, bias=bias)
            self.OW = nn.Linear(context_size, query_size, bias=bias)
            # init
            self._init_weights(self.QW)
            self._init_weights(self.KW)
            self._init_weights(self.VW)
            self._init_weights(self.OW)
        else:
            raise ValueError

        self.scale = self.head_dim**-0.5  # scaling factor

        self.use_norm = norm
        if norm:
            self.norm = nn.LayerNorm(query_size)
            self.norm_context = nn.LayerNorm(context_size)
        self.dropout = nn.Dropout(p=dropout)
        self.attn_dropout = nn.Dropout(p=attention_dropout, inplace=True)

    def _init_weights(self, module: nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # 1: mask / 0: not mask
    ):
        # pre-norm
        if self.use_norm:
            query = self.norm(query)
            context = self.norm_context(context)

        # in projection
        if self.trainable:
            query = self.QW(query)
            key = self.KW(context)
            value = self.VW(context)

        query = einops.rearrange(query, "b i (h d) -> b h i d", h=self.num_heads)
        key = einops.rearrange(key, "b j (h d) -> b h j d", h=self.num_heads)
        value = einops.rearrange(value, "b j (h d) -> b h j d", h=self.num_heads)
        if attention_mask is not None:
            attention_mask = einops.repeat(
                attention_mask, "b i -> b h 1 i", h=self.num_heads
            )

        def scaled_dot_product_attention(query, key, value, mask):
            attn_score = torch.einsum(
                "b h i d, b h j d -> b h i j", query * self.scale, key
            )
            if attention_mask is not None:
                big_neg = -torch.finfo(attn_score.dtype).max
                attn_score = attn_score.masked_fill(mask == 1, big_neg)

            # dropout attn_score
            attn_score = self.attn_dropout(attn_score)

            attn_weights = torch.softmax(attn_score, dim=-1, dtype=torch.float32).to(
                attn_score.dtype
            )  # memory efficiency
            o = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, value)
            return o, attn_weights.detach()

        splited_outputs, attn_weights = scaled_dot_product_attention(
            query, key, value, attention_mask
        )
        outputs = einops.rearrange(splited_outputs, "b h i d -> b i (h d)")
        outputs = self.OW(outputs)

        outputs = self.dropout(outputs)

        if self.return_attention_weights:
            return outputs, attn_weights
        return outputs
