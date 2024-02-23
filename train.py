import gc
import os
import warnings
from dataclasses import asdict
from typing import Any, List

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import wandb
from dotenv import load_dotenv

from src.models import KWComHumor, LightningModel, LMBaseModel
from src.utils import (
    Config,
    LightningDataRetriever,
    create_comet,
    load_dataset,
)

warnings.simplefilter("ignore")


@hydra.main(config_path="conf", config_name="config")
def train(cfg: Config):
    cfg = Config(experiment=cfg.experiment, model=cfg.model, debug=cfg.debug)
    for k_nest, v_nest in asdict(cfg).items():
        print(k_nest)
        if hasattr(v_nest, "items"):
            for k, v in v_nest.items():
                print(f"    {k} = {v}")
        else:
            print(f"    {k_nest} = {v_nest}")
    base = "/".join(os.getcwd().split("/")[:-3])
    # Reproduction
    seed = cfg.experiment.seed
    pl.seed_everything(seed, workers=True)

    # debug
    debug = cfg.debug

    load_dotenv()  # load wandb api key
    wandb.login(key=os.getenv("WANDB_API_KEY"))  # login

    input_dir_base = cfg.experiment.input_dir
    output_dir_base = cfg.experiment.output_dir
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
    print("Input dir :", input_dir)
    print("Output dir:", output_dir)

    traindf = load_dataset(
        data_dir=input_dir,
        task=cfg.experiment.task,
        phase="train",
        debug=debug,
        bs=cfg.experiment.batch_size if not cfg.debug else 2,
        score=cfg.experiment.score,
        undersampling=cfg.experiment.undersampling,
    )
    print(traindf.head())
    validdf = load_dataset(
        data_dir=input_dir,
        task=cfg.experiment.task,
        phase="valid",
        debug=debug,
        bs=cfg.experiment.batch_size if not cfg.debug else 2,
        score=cfg.experiment.score,
    )
    print(validdf.head())
    testdf = load_dataset(
        data_dir=input_dir,
        task=cfg.experiment.task,
        phase="test",
        debug=debug,
        bs=cfg.experiment.batch_size if not cfg.debug else 2,
        score=cfg.experiment.score,
    )
    print(testdf.head())

    # Trainer Arguments
    trainer_args = {
        "max_epochs": cfg.experiment.epoch,
        "deterministic": "warn",
        "profiler": "simple",
        "accumulate_grad_batches": cfg.experiment.accumulate_grad_batches,
        "detect_anomaly": True,
        "gradient_clip_val": cfg.experiment.gradient_clip_val,
    }
    # Device
    if cfg.experiment.device == "gpu":
        trainer_args.update({"gpus": -1})
    elif cfg.experiment.device == "tpu":
        trainer_args.update({"tpu_cores": 8})
    # deterministic

    # Callbacks
    callbacks: List[Any] = []
    # TQDM Callback
    progressbar = pl.callbacks.progress.TQDMProgressBar(
        refresh_rate=20,
    )
    callbacks.append(progressbar)
    # EarlyStopping Callback
    if cfg.experiment.earlystopping:
        es = pl.callbacks.EarlyStopping(
            monitor=cfg.experiment.monitor,
            patience=cfg.experiment.patience,
            mode=cfg.experiment.mode_earlystopping,
        )
        callbacks.append(es)

    log_dir = os.path.join(output_dir, cfg.experiment.name, "logs")
    os.makedirs(log_dir, exist_ok=True)  # mkdir
    logger = pl_loggers.WandbLogger(
        name=cfg.experiment.name,
        save_dir=log_dir,
        project=cfg.experiment.task,
        group=(
            f"exp{cfg.experiment.version:03}"
            if cfg.experiment.version is not None
            else None
        ),
    )

    csv_logger = pl_loggers.CSVLogger(save_dir=log_dir, name="train")
    trainer_args.update({"logger": [logger, csv_logger]})  # register logger

    # Checkpoint Callback
    checkpoint_dir = os.path.join(output_dir, cfg.experiment.name, "weights")
    os.makedirs(checkpoint_dir, exist_ok=True)  # mkdir
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=cfg.experiment.monitor,
        dirpath=checkpoint_dir,
        filename="best",
        # filename="{epoch}-{valid/loss:.3f}-{valid/f1:.3f}",
        # save_top_k=-1,
        mode=cfg.experiment.mode,
    )
    callbacks.append(checkpoint)

    # learning rate callbacks
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    trainer_args.update({"callbacks": callbacks})

    if cfg.experiment.task == "rJokes":
        trainer_args.update({"val_check_interval": 0.2})

    trainer = pl.Trainer(**trainer_args)

    comet_tokenizer = None
    if cfg.experiment.use_keywords:  # create tokenizer and model
        comet_model, comet_tokenizer = create_comet(comet_path, base)

    dm = LightningDataRetriever(
        traindf=traindf,
        validdf=validdf,
        testdf=testdf,
        model_path=cfg.model.model_path,
        seed=seed,
        batch_size=cfg.experiment.batch_size if not cfg.debug else 2,
        max_length=cfg.experiment.max_length,
        label=cfg.experiment.label,
        # max_sample=max_sample,
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
            cache_dir=cfg.experiment.cache_dir,
        )
    else:
        backbone = LMBaseModel(
            model_path=cfg.model.model_path,
            pretrained=cfg.model.pretrained,
            model_type=cfg.model.model_type,
            dropout=cfg.model.dropout,
        )

    wandb.watch(backbone, log="gradients", log_freq=1, log_graph=True)

    model = LightningModel(
        model=backbone,
        lr=cfg.experiment.lr,
        lr_insert=cfg.experiment.lr_insert,
        epoch=cfg.experiment.epoch,
        num_classes=cfg.experiment.num_classes,
        lr_scheduler=cfg.experiment.scheduler,
        warmup_step=cfg.experiment.warmup_step,
        use_keywords=cfg.experiment.use_keywords,
        comet_tokenizer=comet_tokenizer,
        comet_learnable=cfg.experiment.comet_learnable,
        model_path=cfg.model.model_path,
    )

    # Training !!!
    ckpt = cfg.experiment.ckpt
    print("\n===START  TRAINING===\n")
    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt)
    print("\n===FINISH TRAINING===\n")

    del dm, model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    train()
