import sys
from dataclasses import dataclass
from typing import List, Optional, Union

if sys.version_info > (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


@dataclass
class Experiment:
    version: int = 0
    name: str = "debug"
    task: Literal["Humicroedit", "HaHackathon"] = "HaHackathon"
    seed: int = 42
    input_dir: str = "data/processed"
    output_dir: str = "outputs"
    label: str = "label"
    input_method: Optional[Literal["head", "tail", "head-tail"]] = "head-tail"
    head_ratio: float = 0.75
    use_keywords: bool = False
    comet_path: str = "pretrained/conceptnet_pretrained_model.pickle"
    relation: Union[List[str], str] = "all"
    max_keyword_num: int = 6
    comet_learnable: bool = False
    batch_size: int = 32
    max_length: int = 128
    num_classes: int = 2
    lr: float = 2e-5
    lr_insert: float = 1e-4
    epoch: int = 15
    device: Literal["cpu", "gpu", "tpu"] = "cpu"
    monitor: str = "valid/loss"
    mode: str = "min"
    accumulate_grad_batches: int = 1
    earlystopping: bool = True
    patience: int = 3
    mode_earlystopping: str = "min"
    gradient_clip_val: Optional[Union[int, float]] = None
    ckpt: Optional[str] = None
    score: Optional[int] = None
    undersampling: bool = False
    scheduler: bool = False
    warmup_step: int = 0
    cache_dir: Optional[str] = None


@dataclass
class Model:
    model_path: str = "bert-base-uncased"
    pretrained: bool = True
    model_type: Optional[str] = None
    num_heads: int = 8
    dropout: float = 0.4
    fusion_dropout_prob: float = 0.0
    keywords_dropout_prob: float = 0.0
    norm: bool = False


@dataclass
class Config:
    experiment: Experiment
    model: Model
    debug: bool

    def __post_init__(self):
        object.__setattr__(self, "experiment", Experiment(**self.experiment))
        object.__setattr__(self, "model", Model(**self.model))
