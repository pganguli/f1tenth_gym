#!/usr/bin/env python3
"""
Advanced training script for F1Tenth LiDAR DNN models using PyTorch Lightning.
Supports YAML configuration, advanced dataloading, and automatic tuning.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from f110_planning.utils.nn_models import get_architecture

# Suppress library-level deprecation warnings for cleaner output
warnings.filterwarnings("ignore", message=".*treespec.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightning")


class LidarDataset(Dataset):
    """
    Advanced LiDAR Dataset with memory-resident caching.
    """

    def __init__(self, data_path: str, target_col: str):
        # Caching: Load once into memory
        data = np.load(data_path)
        self.x = data["scans"].astype(np.float32)

        # Handle multi-target columns (e.g., "left_dist,right_dist")
        if "," in target_col:
            cols = [c.strip() for c in target_col.split(",")]
            self.y = np.stack([data[c].astype(np.float32) for c in cols], axis=1)
        else:
            self.y = data[target_col].astype(np.float32).reshape(-1, 1)

        # Consistent normalization
        self.x = np.clip(self.x / 10.0, 0, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]).unsqueeze(0), torch.from_numpy(self.y[idx])


class LidarDataModule(L.LightningDataModule):
    """
    Lightning DataModule with multi-worker, prefetching, and pinning.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config["data"]
        # Attributes reduced to stay below limit
        self.train_path = self.cfg["train_path"]
        self.target_col = self.cfg["target_col"]
        self.batch_size = self.cfg["batch_size"]

        # Initializing datasets to None to satisfy W0201
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation."""
        full_dataset = LidarDataset(self.train_path, self.target_col)

        # Dataset Sharding
        val_split = self.cfg.get("val_split", 0.2)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is None. Call setup() first.")
        num_workers = self.cfg.get("num_workers", 4)
        # Use pin_memory only if GPU is available to avoid warnings
        pin_memory = self.cfg.get("pin_memory", True) and torch.cuda.is_available()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is None. Call setup() first.")
        num_workers = self.cfg.get("num_workers", 4)
        pin_memory = self.cfg.get("pin_memory", True) and torch.cuda.is_available()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.cfg.get("prefetch_factor", 2) if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )


class LidarLightningModule(L.LightningModule):
    """
    LightningModule with ReduceLROnPlateau.
    """

    def __init__(
        self,
        arch_id: int,
        task: str = "heading",
        **kwargs,
    ):
        super().__init__()
        # Group hypers to reduce instance attributes and satisfies too-many-args
        self.save_hyperparameters()

        self.model = get_architecture(arch_id, task=task)
        self.criterion = nn.MSELoss()

        # Added for advanced logging
        self.example_input_array = torch.randn(1, 1, 1080)
        self.train_step_count = 0

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        x = args[0]
        return self.model(x)

    def training_step(self, *args, **kwargs):
        """Execute a training step with enhanced logging."""
        batch = args[0]
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        # ---- CORE LOSSES ----
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # ---- LR LOGGING ----
        opt = self.optimizers()
        # Handle case where self.optimizers() might be a list or a single optimizer
        if isinstance(opt, list):
            lr = opt[0].param_groups[0]["lr"]
        else:
            lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=False)

        self.train_step_count += 1
        return loss

    def on_after_backward(self):
        """Log gradient norms for health monitoring."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    def validation_step(self, *args, **kwargs):
        """Execute a validation step with MAE tracking."""
        batch = args[0]
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)

        # Mean Absolute Error is helpful for distance/angle regression
        mae = torch.mean(torch.abs(y_hat - y))
        self.log("val/mae", mae, prog_bar=False, on_epoch=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log system metrics."""
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            self.log("sys/gpu_mem_gb", mem, on_step=True, prog_bar=False)

    def on_train_epoch_end(self):
        """Log weight histograms to debug filter health."""
        if self.logger and hasattr(self.logger, "experiment"):
            tb_logger = self.logger.experiment
            # Check if it has add_histogram (TensorBoard specific)
            if hasattr(tb_logger, "add_histogram"):
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        tb_logger.add_histogram(
                            f"weights/{name}", param, self.current_epoch
                        )

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=self.hparams.lr_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def run_single_training(config: dict, arch_id: int):
    """Run a single training session for a specific architecture."""
    # Set matmul precision for better GPU utilization and to suppress warnings
    torch.set_float32_matmul_precision("high")

    L.seed_everything(42)

    # Setup Model
    task = "wall" if "," in config["data"]["target_col"] else "heading"
    module = LidarLightningModule(
        arch_id=arch_id,
        task=task,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        lr_patience=config["training"]["lr_patience"],
    )

    # Logger
    model_name = f"{config['data']['target_col']}_arch{arch_id}"
    logger = TensorBoardLogger("lightning_logs", name=model_name)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"data/models/checkpoints/{model_name}",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="auto",
        devices=1,
        precision=config["training"].get("precision", "32")
        if torch.cuda.is_available()
        else "32",
        logger=logger,
        callbacks=callbacks,
        profiler=config["training"].get("profiler", None),
    )

    # Data
    dm = LidarDataModule(config)

    # 1. Automatic LR Finding
    if config["training"].get("auto_lr_find", False):
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module, datamodule=dm)
        if lr_finder:
            suggested_lr = lr_finder.suggestion()
            if suggested_lr:
                print(f"Best LR found: {suggested_lr}")
                module.hparams.lr = suggested_lr

    # 2. Resume Support
    ckpt_path = None
    if config["training"].get("resume", True):
        last_ckpt = Path(f"data/models/checkpoints/{model_name}/last.ckpt")
        if last_ckpt.exists():
            print(f"Resuming from {last_ckpt}")
            ckpt_path = str(last_ckpt)

    # 3. Training
    trainer.fit(module, datamodule=dm, ckpt_path=ckpt_path)

    # Save final weights
    out_path = Path("data/models")
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(module.model.state_dict(), out_path / f"{model_name}.pth")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="scripts/train/config_heading.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Hyperparameter Search across architectures if specified
    arch_ids = config.get("model", {}).get("arch_ids", [config["model"]["arch_id"]])

    for aid in arch_ids:
        print(f"\n--- Starting training for Architecture {aid} ---")
        run_single_training(config, aid)


if __name__ == "__main__":
    main()
