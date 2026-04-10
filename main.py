"""
main.py — Master orchestration script for MRI Super Resolution.

Runs the full pipeline:
  Phase 1: Data preparation & sanity check
  Phase 2: Train SRCNN
  Phase 3: Train U-Net SR
  Phase 4: Evaluate all methods (Bicubic, SRCNN, U-Net)
  Phase 5: Generate all visualizations
"""

import random
import sys
import time

import numpy as np
import torch

import config as cfg


def set_seed():
    """Set all random seeds for reproducibility."""
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)


def main():
    total_start = time.time()
    set_seed()

    print("=" * 60)
    print("  MRI Super Resolution Pipeline")
    print(f"  Device: {cfg.DEVICE}")
    print(f"  Seed:   {cfg.SEED}")
    print("=" * 60)

    # ================================================================
    # Phase 1: Data preparation
    # ================================================================
    print("\n▸ Phase 1: Loading data …")
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")
    print(f"  Test  batches: {len(test_loader)}")

    # Quick sanity check
    lr, hr = next(iter(train_loader))
    assert lr.shape[1] == 1 and hr.shape[1] == 1, "Expected single channel"
    print(f"  Batch shape — LR: {lr.shape}, HR: {hr.shape}")
    print("  ✓ Data loaded successfully.\n")

    # ================================================================
    # Phase 2: Train SRCNN
    # ================================================================
    print("▸ Phase 2: Training SRCNN …")
    from train import train_model
    from models import SRCNN, UNetSR

    srcnn = SRCNN()
    train_model(
        model=srcnn,
        model_name="SRCNN",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=str(cfg.SRCNN_CKPT),
    )

    # ================================================================
    # Phase 3: Train U-Net SR
    # ================================================================
    print("\n▸ Phase 3: Training U-Net SR …")
    unet = UNetSR()
    train_model(
        model=unet,
        model_name="UNet",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=str(cfg.UNET_CKPT),
    )

    # ================================================================
    # Phase 4: Evaluate all methods
    # ================================================================
    print("\n▸ Phase 4: Evaluating all methods …")
    from evaluate import main as evaluate_main
    evaluate_main()

    # ================================================================
    # Phase 5: Generate visualizations
    # ================================================================
    print("\n▸ Phase 5: Generating visualizations …")
    from visualize import plot_loss_curves, plot_comparison_grid, plot_error_maps

    plot_loss_curves()
    plot_comparison_grid(val_loader, n_samples=5)
    plot_error_maps(val_loader, n_samples=3)

    # ================================================================
    # Done
    # ================================================================
    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline complete!  Total time: {elapsed / 60:.1f} min")
    print(f"  Outputs saved to: {cfg.OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
