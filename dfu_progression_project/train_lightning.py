import json
import os
from os.path import join

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig

from src.data_module import DFUDataModule
from src.model import DFUModel
from src.utils import generate_run_id


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    run_id = generate_run_id()

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    dm = DFUDataModule(
        train_csv_path=cfg.train_csv_path,
        val_csv_path=cfg.val_csv_path,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_class_weighting=cfg.use_class_weighting,
        use_weighted_sampler=cfg.use_weighted_sampler,
    )
    dm.setup()

    model = DFUModel(
        num_classes=dm.num_classes,
        model_name=cfg.model_name,
        learning_rate=cfg.learning_rate,
        class_weights=dm.class_weights,
        use_scheduler=cfg.use_scheduler,
    )

    tb_logger = TensorBoardLogger(save_dir=cfg.logs_dir, name="", version=run_id)
    csv_logger = CSVLogger(save_dir=cfg.logs_dir, name="", version=run_id)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        dirpath=join(cfg.checkpoint_dirpath, run_id),
        filename="{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}-{val_kappa:.2f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=5,
    )

    trainer.fit(model, dm)

    metric_summary = {}
    for key, value in trainer.callback_metrics.items():
        if hasattr(value, "item"):
            metric_summary[key] = float(value.item())
        else:
            metric_summary[key] = float(value)

    metric_summary["run_id"] = run_id
    metric_summary["best_model_path"] = checkpoint_callback.best_model_path
    metric_summary["best_model_score"] = (
        float(checkpoint_callback.best_model_score.item())
        if checkpoint_callback.best_model_score is not None
        else None
    )

    metrics_dir = getattr(cfg, "metrics_dir", "artifacts/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = join(metrics_dir, f"{run_id}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metric_summary, f, indent=2)

    print(f"Saved metrics summary to: {summary_path}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
