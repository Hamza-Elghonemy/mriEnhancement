"""
train.py — Training loop for SRCNN and U-Net SR models.

Trains each model with L1 loss, Adam optimiser and StepLR scheduler.
Saves the best checkpoint (by validation loss) and exports a per-epoch
loss log for later visualisation.
"""

import os
import json
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import config as cfg
from dataset import get_dataloaders
from models import SRCNN, UNetSR


# ============================================================================
# Training & validation steps
# ============================================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    """Run one training epoch.  Returns average batch loss."""
    model.train()
    running_loss = 0.0
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        pred = model(lr_batch)
        loss = criterion(pred, hr_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * lr_batch.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> float:
    """Run validation.  Returns average loss."""
    model.eval()
    running_loss = 0.0
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        pred = model(lr_batch)
        loss = criterion(pred, hr_batch)
        running_loss += loss.item() * lr_batch.size(0)
    return running_loss / len(loader.dataset)


# ============================================================================
# Full training pipeline
# ============================================================================

def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: str,
    device: torch.device = cfg.DEVICE,
    num_epochs: int = cfg.NUM_EPOCHS,
    lr: float = cfg.LEARNING_RATE,
) -> Dict[str, List[float]]:
    """Train a model end-to-end.

    Parameters
    ----------
    model : nn.Module
    model_name : str  – for logging (e.g. "SRCNN", "UNet")
    train_loader, val_loader : DataLoaders
    checkpoint_path : str – where to save the best model
    device, num_epochs, lr : training config

    Returns
    -------
    log : dict with keys "train_loss" and "val_loss", each a list[float].
    """
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    best_val = float("inf")
    log: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name}  |  device={device}  |  epochs={num_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        marker = " ★" if val_loss < best_val else ""
        print(f"  [{model_name}] Epoch {epoch:3d}/{num_epochs} | "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={lr_now:.1e}  ({dt:.1f}s){marker}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"  [{model_name}] Best val loss: {best_val:.5f}  →  {checkpoint_path}")

    # Save loss log as JSON for visualise.py
    log_path = str(checkpoint_path).replace(".pth", "_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f)
    print(f"  [{model_name}] Loss log saved to {log_path}")

    return log


# ============================================================================
# Main entry-point: train both models sequentially
# ============================================================================

def main():
    # Reproducibility
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    # Build data loaders
    train_loader, val_loader, _ = get_dataloaders()

    # --- Train SRCNN ---
    srcnn = SRCNN()
    srcnn_log = train_model(
        model=srcnn,
        model_name="SRCNN",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=str(cfg.SRCNN_CKPT),
    )

    # --- Train U-Net SR ---
    unet = UNetSR()
    unet_log = train_model(
        model=unet,
        model_name="UNet",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=str(cfg.UNET_CKPT),
    )

    print("\n✓ Training complete for both models.")


if __name__ == "__main__":
    main()
