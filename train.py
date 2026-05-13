"""
train.py — Training loop for SRCNN and U-Net SR models.

Trains each model with L1 loss, Adam optimiser and StepLR scheduler.
Optionally adds perceptual loss and Rician noise augmentation.
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
from models import SRCNN, UNetSR, PerceptualLoss, add_rician_noise


# ============================================================================
# Training & validation steps
# ============================================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    l1_criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    perceptual_criterion: Optional[nn.Module] = None,
                    perceptual_weight: float = 0.0) -> Dict[str, float]:
    """Run one training epoch.  Returns averaged loss components."""
    model.train()
    running_total = 0.0
    running_l1 = 0.0
    running_perceptual = 0.0
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        if cfg.USE_RICIAN_NOISE and random.random() < cfg.RICIAN_PROB:
            lr_batch = add_rician_noise(lr_batch, sigma=cfg.RICIAN_SIGMA)

        pred = model(lr_batch)
        l1_loss = l1_criterion(pred, hr_batch)
        total_loss = l1_loss
        perceptual_loss = None
        if perceptual_criterion is not None:
            perceptual_loss = perceptual_criterion(pred, hr_batch)
            total_loss = l1_loss + perceptual_weight * perceptual_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_size = lr_batch.size(0)
        running_total += total_loss.item() * batch_size
        running_l1 += l1_loss.item() * batch_size
        if perceptual_loss is not None:
            running_perceptual += perceptual_loss.item() * batch_size

    dataset_size = len(loader.dataset)
    stats = {
        "total": running_total / dataset_size,
        "l1": running_l1 / dataset_size,
    }
    if perceptual_criterion is not None:
        stats["perceptual"] = running_perceptual / dataset_size
    return stats


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             l1_criterion: nn.Module,
             device: torch.device,
             perceptual_criterion: Optional[nn.Module] = None,
             perceptual_weight: float = 0.0) -> Dict[str, float]:
    """Run validation.  Returns averaged loss components."""
    model.eval()
    running_total = 0.0
    running_l1 = 0.0
    running_perceptual = 0.0
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        pred = model(lr_batch)
        l1_loss = l1_criterion(pred, hr_batch)
        total_loss = l1_loss
        perceptual_loss = None
        if perceptual_criterion is not None:
            perceptual_loss = perceptual_criterion(pred, hr_batch)
            total_loss = l1_loss + perceptual_weight * perceptual_loss

        batch_size = lr_batch.size(0)
        running_total += total_loss.item() * batch_size
        running_l1 += l1_loss.item() * batch_size
        if perceptual_loss is not None:
            running_perceptual += perceptual_loss.item() * batch_size

    dataset_size = len(loader.dataset)
    stats = {
        "total": running_total / dataset_size,
        "l1": running_l1 / dataset_size,
    }
    if perceptual_criterion is not None:
        stats["perceptual"] = running_perceptual / dataset_size
    return stats


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
    l1_criterion = nn.L1Loss()
    perceptual_criterion = None
    if cfg.USE_PERCEPTUAL_LOSS:
        perceptual_criterion = PerceptualLoss().to(device)
        perceptual_criterion.eval()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    best_val = float("inf")
    log: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_l1": [],
        "val_l1": [],
    }
    if perceptual_criterion is not None:
        log["train_perceptual"] = []
        log["val_perceptual"] = []

    print(f"\n{'='*60}")
    print(f"  Training {model_name}  |  device={device}  |  epochs={num_epochs}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            l1_criterion,
            optimizer,
            device,
            perceptual_criterion=perceptual_criterion,
            perceptual_weight=cfg.PERCEPTUAL_WEIGHT,
        )
        val_stats = validate(
            model,
            val_loader,
            l1_criterion,
            device,
            perceptual_criterion=perceptual_criterion,
            perceptual_weight=cfg.PERCEPTUAL_WEIGHT,
        )
        scheduler.step()

        log["train_loss"].append(train_stats["total"])
        log["val_loss"].append(val_stats["total"])
        log["train_l1"].append(train_stats["l1"])
        log["val_l1"].append(val_stats["l1"])
        if perceptual_criterion is not None:
            log["train_perceptual"].append(train_stats["perceptual"])
            log["val_perceptual"].append(val_stats["perceptual"])

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        marker = " ★" if val_stats["total"] < best_val else ""
        extra = ""
        if perceptual_criterion is not None:
            extra = (f" l1={train_stats['l1']:.5f} "
                     f"ploss={train_stats['perceptual']:.5f}")
        print(f"  [{model_name}] Epoch {epoch:3d}/{num_epochs} | "
              f"train={train_stats['total']:.5f}  val={val_stats['total']:.5f}"
              f"{extra}  lr={lr_now:.1e}  ({dt:.1f}s){marker}")

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
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
