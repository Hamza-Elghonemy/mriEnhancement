"""
visualize.py — Visualization outputs for MRI Super Resolution.

Generates:
  1. loss_curves.png       – Training & validation loss per epoch (SRCNN + U-Net)
  2. comparison_grid.png   – 5-column side-by-side using synthetic val set
  3. error_maps.png        – Per-pixel absolute difference heatmaps with annotations
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

import config as cfg
from dataset import get_dataloaders
from models import SRCNN, UNetSR


# ============================================================================
# 1. Loss curves  (FR-06.1)
# ============================================================================

def plot_loss_curves(srcnn_log_path: Optional[str] = None,
                     unet_log_path: Optional[str] = None,
                     save_path: Optional[str] = None):
    """Plot training and validation L1 loss per epoch for both models."""

    if srcnn_log_path is None:
        srcnn_log_path = str(cfg.SRCNN_CKPT).replace(".pth", "_log.json")
    if unet_log_path is None:
        unet_log_path = str(cfg.UNET_CKPT).replace(".pth", "_log.json")
    if save_path is None:
        save_path = str(cfg.FIGURES_DIR / "loss_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, log_path, name, color in [
        (axes[0], srcnn_log_path, "SRCNN", ("#2196F3", "#FF9800")),
        (axes[1], unet_log_path,  "U-Net SR", ("#4CAF50", "#E91E63")),
    ]:
        with open(log_path, "r") as f:
            log = json.load(f)
        epochs = range(1, len(log["train_loss"]) + 1)
        ax.plot(epochs, log["train_loss"], label="Train", color=color[0],
                linewidth=2, alpha=0.9)
        ax.plot(epochs, log["val_loss"], label="Validation", color=color[1],
                linewidth=2, alpha=0.9, linestyle="--")
        ax.set_title(f"{name} — L1 Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L1 Loss")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(log["train_loss"]))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Loss curves saved → {save_path}")


# ============================================================================
# Helper: load both trained models
# ============================================================================

def _load_models(device: torch.device = cfg.DEVICE):
    """Load SRCNN and U-Net SR from checkpoints."""
    srcnn = SRCNN()
    srcnn.load_state_dict(torch.load(str(cfg.SRCNN_CKPT), map_location=device,
                                     weights_only=True))
    srcnn.to(device).eval()

    unet = UNetSR()
    unet.load_state_dict(torch.load(str(cfg.UNET_CKPT), map_location=device,
                                    weights_only=True))
    unet.to(device).eval()

    return srcnn, unet


def _collect_samples(loader: DataLoader):
    """Collect all (LR, HR) tensors from a DataLoader into lists."""
    all_lr, all_hr = [], []
    for lr_b, hr_b in loader:
        for i in range(lr_b.shape[0]):
            all_lr.append(lr_b[i])
            all_hr.append(hr_b[i])
    return all_lr, all_hr


# ============================================================================
# 2. Comparison grid  (FR-06.2) — uses synthetic val set
# ============================================================================

@torch.no_grad()
def plot_comparison_grid(val_loader: DataLoader,
                         n_samples: int = 5,
                         save_path: Optional[str] = None):
    """5-column panel: LR Input → SRCNN → U-Net → HR Ground Truth → Error.

    Uses the validation set (synthetic LR/HR pairs from the same 64mT
    images) so that HR is the true target and errors are meaningful.
    """
    if save_path is None:
        save_path = str(cfg.FIGURES_DIR / "comparison_grid.png")

    device = cfg.DEVICE
    srcnn, unet = _load_models(device)

    all_lr, all_hr = _collect_samples(val_loader)

    # Pick evenly-spaced indices
    total = len(all_lr)
    indices = np.linspace(0, total - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 5, figsize=(22, 4.2 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["LR Input (Bicubic↑)", "SRCNN", "U-Net SR",
                  "HR Ground Truth", "|HR − U-Net| Error"]

    for row, idx in enumerate(indices):
        lr_t = all_lr[idx].unsqueeze(0).to(device)
        hr_np = all_hr[idx][0].numpy()
        lr_np = all_lr[idx][0].numpy()

        srcnn_pred = np.clip(srcnn(lr_t).cpu().numpy()[0, 0], 0, 1)
        unet_pred  = np.clip(unet(lr_t).cpu().numpy()[0, 0], 0, 1)
        error_map  = np.abs(hr_np - unet_pred)

        panels = [lr_np, srcnn_pred, unet_pred, hr_np, error_map]
        cmaps  = ["gray", "gray", "gray", "gray", "hot"]

        for col, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row, col]
            im = ax.imshow(panel, cmap=cmap, vmin=0,
                           vmax=1 if col < 4 else max(error_map.max(), 0.01))
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold")
            if col == 4:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Super Resolution Comparison — Synthetic Validation Set",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Comparison grid saved → {save_path}")


# ============================================================================
# 3. Error maps with anatomical annotations  (FR-06.4)
# ============================================================================

@torch.no_grad()
def plot_error_maps(val_loader: DataLoader,
                    n_samples: int = 3,
                    save_path: Optional[str] = None):
    """Per-pixel |HR − Prediction| heatmaps with annotated high-error regions.

    Uses synthetic val set for meaningful error analysis.
    """
    if save_path is None:
        save_path = str(cfg.FIGURES_DIR / "error_maps.png")

    device = cfg.DEVICE
    srcnn, unet = _load_models(device)

    all_lr, all_hr = _collect_samples(val_loader)

    total = len(all_lr)
    indices = np.linspace(0, total - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 5.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["HR Ground Truth", "SRCNN |Error|", "U-Net |Error|"]

    for row, idx in enumerate(indices):
        lr_t = all_lr[idx].unsqueeze(0).to(device)
        hr_np = all_hr[idx][0].numpy()

        srcnn_pred = np.clip(srcnn(lr_t).cpu().numpy()[0, 0], 0, 1)
        unet_pred  = np.clip(unet(lr_t).cpu().numpy()[0, 0], 0, 1)

        err_srcnn = np.abs(hr_np - srcnn_pred)
        err_unet  = np.abs(hr_np - unet_pred)

        # Panel 0: HR reference
        axes[row, 0].imshow(hr_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(col_titles[0] if row == 0 else "", fontsize=12,
                               fontweight="bold")
        axes[row, 0].axis("off")

        # Panel 1: SRCNN error
        vmax_err = max(err_srcnn.max(), err_unet.max(), 0.01)
        im1 = axes[row, 1].imshow(err_srcnn, cmap="hot", vmin=0, vmax=vmax_err)
        axes[row, 1].set_title(col_titles[1] if row == 0 else "", fontsize=12,
                               fontweight="bold")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Panel 2: U-Net error
        im2 = axes[row, 2].imshow(err_unet, cmap="hot", vmin=0, vmax=vmax_err)
        axes[row, 2].set_title(col_titles[2] if row == 0 else "", fontsize=12,
                               fontweight="bold")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # --- Annotate high-error anatomical regions on U-Net error map ---
        threshold = np.percentile(err_unet, 95)
        high_err = err_unet > threshold
        if high_err.any():
            ys, xs = np.where(high_err)
            cy, cx = int(ys.mean()), int(xs.mean())
            h, w = err_unet.shape

            annotations = []
            if cy < h * 0.35:
                annotations.append(("Cortical surface / Sulci", cx, cy))
            elif cy > h * 0.65:
                annotations.append(("Cerebellum / Brainstem", cx, cy))
            else:
                annotations.append(("WM/GM boundary", cx, cy))

            if len(ys) > 10:
                left_mask  = xs < w // 2
                right_mask = xs >= w // 2
                if left_mask.any() and right_mask.any():
                    cy2 = int(ys[right_mask].mean())
                    cx2 = int(xs[right_mask].mean())
                    if abs(cy2 - cy) > 15 or abs(cx2 - cx) > 15:
                        annotations.append(("Ventricular boundary", cx2, cy2))

            for label, ax_x, ax_y in annotations[:2]:
                axes[row, 2].annotate(
                    label,
                    xy=(ax_x, ax_y),
                    xytext=(ax_x + 20, ax_y - 25),
                    fontsize=9, color="cyan", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                )

    plt.suptitle("Per-Pixel Error Maps — Synthetic Validation Set",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Error maps saved → {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    _, val_loader, _ = get_dataloaders()

    plot_loss_curves()
    plot_comparison_grid(val_loader, n_samples=5)
    plot_error_maps(val_loader, n_samples=3)

    print("\n✓ All visualizations generated.")


if __name__ == "__main__":
    main()
