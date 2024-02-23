import gc
import os
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch

from src.models import KWComHumor, LightningModel, LMBaseModel
from src.utils import (
    Config,
    LightningDataRetriever,
    create_comet,
    load_dataset,
)

warnings.simplefilter("ignore")


@hydra.main(config_path="conf", config_name="config")
def test(cfg: Config):
    cfg = Config(experiment=cfg.experiment, model=cfg.model, debug=cfg.debug)
    base = "/".join(os.getcwd().split("/")[:-3])
    # Reproduction
    seed = cfg.experiment.seed
    pl.seed_everything(seed, workers=True)

    input_dir_base = cfg.experiment.input_dir
    output_dir_base = cfg.experiment.output_dir
    # overwrite settings
    if cfg.experiment.task in ("Humicroedit"):
        input_dir = os.path.join(
            input_dir_base, "semeval-2020-task-7-dataset", "subtask-1"
        )
    else:
        input_dir = os.path.join(input_dir_base, cfg.experiment.task)
    output_dir = os.path.join(
        output_dir_base, cfg.experiment.task, f"exp{cfg.experiment.version:03}"
    )

    input_dir = os.path.join(base, input_dir)
    output_dir = os.path.join(base, output_dir)
    comet_path = os.path.join(base, cfg.experiment.comet_path)

    traindf = load_dataset(
        data_dir=input_dir,
        task=cfg.experiment.task,
        phase="train",
        score=cfg.experiment.score,
    )
    validdf = load_dataset(
        data_dir=input_dir,
        task=cfg.experiment.task,
        phase="valid",
        score=cfg.experiment.score,
    )
    testdf = load_dataset(
        data_dir=input_dir,
        task=cfg.experiment.task,
        phase="test",
        score=cfg.experiment.score,
    )
    print(testdf.head())

    # Trainer Arguments
    trainer_args = {
        "max_epochs": cfg.experiment.epoch,
        "deterministic": "warn",
        "profiler": "simple",
    }
    # Device
    if cfg.experiment.device == "gpu":
        trainer_args.update({"gpus": -1})
    elif cfg.experiment.device == "tpu":
        trainer_args.update({"tpu_cores": 8})

    # Callbacks
    callbacks = []
    # TQDM Callback
    progressbar = pl.callbacks.progress.TQDMProgressBar(
        refresh_rate=20,
    )
    callbacks.append(progressbar)

    # Logger
    log_dir = os.path.join(output_dir, cfg.experiment.name, "logs")
    os.makedirs(log_dir, exist_ok=True)  # mkdir
    logger = pl_loggers.CSVLogger(save_dir=log_dir, name="test")
    trainer_args.update({"logger": logger})

    trainer_args.update({"callbacks": callbacks})

    trainer = pl.Trainer(**trainer_args)

    # Create DataModule
    comet_tokenizer = None
    if cfg.experiment.use_keywords:  # create tokenizer and model
        comet_model, comet_tokenizer = create_comet(comet_path, base)
    dm = LightningDataRetriever(
        traindf=traindf,
        validdf=validdf,
        testdf=testdf,
        model_path=cfg.model.model_path,
        seed=seed,
        batch_size=cfg.experiment.batch_size,
        max_length=cfg.experiment.max_length,
        label=cfg.experiment.label,
        use_keywords=cfg.experiment.use_keywords,
        comet_tokenizer=comet_tokenizer,
        relation=cfg.experiment.relation,
        input_method=cfg.experiment.input_method,
        head_ratio=cfg.experiment.head_ratio,
        max_keyword_num=cfg.experiment.max_keyword_num,
    )

    print("=== MODEL SETTINGS ===")
    if cfg.experiment.use_keywords:
        model_settings = "Commonsense-aware Attentive Model"
    else:
        model_settings = "PMLM-only baseline"
    print(model_settings)
    print("=== MODEL SETTINGS ===")

    if cfg.experiment.use_keywords:
        backbone = KWComHumor(
            model_path_lm=cfg.model.model_path,
            comet_model=comet_model,
            model_type_lm=cfg.model.model_type,
            num_label=cfg.experiment.num_classes,
            vocab_size=len(comet_tokenizer.encoder),
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            fusion_dropout_prob=cfg.model.fusion_dropout_prob,
            keywords_dropout_prob=cfg.model.keywords_dropout_prob,
            norm=cfg.model.norm,
        )
    else:
        backbone = LMBaseModel(
            model_path=cfg.model.model_path,
            pretrained=cfg.model.pretrained,
            model_type=cfg.model.model_type,
            dropout=cfg.model.dropout,
        )

    # Loading Checkpoint
    ckpt = None
    if os.path.isfile(
        os.path.join(
            output_dir,
            cfg.experiment.name,
            "weights",
            "best.ckpt",
        )
    ):
        ckpt = os.path.join(output_dir, cfg.experiment.name, "weights", "best.ckpt")
    top: Path = Path(output_dir) / (cfg.experiment.name) / "weights"
    weight = sorted([path for path in top.iterdir()])[0]
    ckpt = weight.as_posix()

    print(ckpt)
    if ckpt is None:
        model = LightningModel(
            model=backbone,
            lr=cfg.experiment.lr,
            lr_insert=cfg.experiment.lr_insert,
            epoch=cfg.experiment.epoch,
            num_classes=cfg.experiment.num_classes,
            use_keywords=cfg.experiment.use_keywords,
            comet_tokenizer=comet_tokenizer,
            comet_learnable=cfg.experiment.comet_learnable,
        )
    else:
        model = LightningModel.load_from_checkpoint(
            ckpt,
            model=backbone,
            lr=cfg.experiment.lr,
            lr_insert=cfg.experiment.lr_insert,
            epoch=cfg.experiment.epoch,
            num_classes=cfg.experiment.num_classes,
            use_keywords=cfg.experiment.use_keywords,
            comet_tokenizer=comet_tokenizer,
            comet_learnable=cfg.experiment.comet_learnable,
        )

    # Test !!!
    print("\n===START TEST===\n")
    trainer.test(model=model, datamodule=dm)
    print("\n===FINISH TEST===\n")
    # Showing Trainig/Validation
    del dm, model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test()
