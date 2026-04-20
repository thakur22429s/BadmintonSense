"""Training loop with early stopping, LR scheduling, and checkpointing."""

from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from src.training.losses import LabelSmoothingCrossEntropy
from src.training.evaluator import evaluate_model


class Trainer:
    """Handles the full training loop for stroke classification models."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        model_type: str = "lstm",
        device: torch.device | None = None,
        save_dir: Path | None = None,
    ):
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = device or torch.device("cpu")
        self.save_dir = save_dir or Path(config["paths"]["models"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)

        # Training config
        train_cfg = config["training"][model_type]
        self.max_epochs = config["training"]["max_epochs"]
        self.patience = config["training"]["early_stopping_patience"]

        # Loss
        self.criterion = LabelSmoothingCrossEntropy(
            num_classes=config["classes"]["num_classes"],
            smoothing=config["training"]["label_smoothing"],
        )

        # Optimizer
        if train_cfg["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg["learning_rate"],
                weight_decay=train_cfg.get("weight_decay", 0),
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg["learning_rate"],
            )

        # LR scheduler
        if train_cfg["lr_scheduler"] == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=train_cfg["lr_patience"],
                factor=train_cfg["lr_factor"],
            )
        elif train_cfg["lr_scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs - train_cfg.get("warmup_epochs", 0),
            )
        else:
            self.scheduler = None

        self.history = {"train_loss": [], "val_loss": [], "val_f1": [], "lr": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_id: str = "default",
    ) -> dict:
        """Run full training loop.

        Returns:
            Training history dict.
        """
        best_f1 = 0.0
        patience_counter = 0
        best_model_path = self.save_dir / f"best_{self.model_type}_{fold_id}.pt"

        class_names = self.config["classes"]["names"]

        for epoch in range(self.max_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_results = evaluate_model(self.model, val_loader, self.device, class_names)
            val_f1 = val_results["macro_f1"]

            # Compute val loss
            val_loss = self._compute_loss(val_loader)

            # LR scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_f1)
            elif self.scheduler is not None:
                self.scheduler.step()

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            print(
                f"Epoch {epoch+1:3d}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "config": self.config,
                }, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1} (best F1: {best_f1:.4f})")
                    break

        # Load best model
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} (F1: {best_f1:.4f})")

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_loss(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)
