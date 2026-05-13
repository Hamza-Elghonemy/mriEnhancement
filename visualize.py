"""
visualize.py — Visualization outputs for MRI Super Resolution.

Generates:
  1. loss_curves.png       – Training & validation loss per epoch (SRCNN + U-Net)
  2. comparison_grid.png   – 5-column side-by-side using synthetic val set
    3. error_maps.png        – Per-pixel absolute difference heatmaps with annotations
    4. triplet_comparison.png – Raw 64mT vs synthetic 4x LR vs paired 3T reference
    5. perceptual_loss_curves.png – Total, L1, and perceptual loss curves
    6. rician_noise_examples.png  – Raw vs noisy vs difference panel
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
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader

import config as cfg
from dataset import get_dataloaders
from dataset import bicubic_downsample, bicubic_upsample_np, extract_2d_slices
from dataset import resize_slice, scan_bids_directory
from models import SRCNN, UNetSR, add_rician_noise


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
        for step_epoch in (cfg.LR_STEP_SIZE, cfg.LR_STEP_SIZE * 2):
            if step_epoch <= len(log["train_loss"]):
                ax.axvline(step_epoch, color="#444444", linestyle="--",
                           linewidth=1.2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Loss curves saved → {save_path}")


def _load_loss_log(log_path: str) -> Dict[str, List[float]]:
    with open(log_path, "r") as f:
        return json.load(f)


def plot_perceptual_loss_curves(srcnn_log_path: Optional[str] = None,
                                unet_log_path: Optional[str] = None,
                                save_path: Optional[str] = None):
    """Plot total, L1, and perceptual loss curves if present in logs."""
    if srcnn_log_path is None:
        srcnn_log_path = str(cfg.SRCNN_CKPT).replace(".pth", "_log.json")
    if unet_log_path is None:
        unet_log_path = str(cfg.UNET_CKPT).replace(".pth", "_log.json")
    if save_path is None:
        save_path = str(cfg.FIGURES_DIR / "perceptual_loss_curves.png")

    srcnn_log = _load_loss_log(srcnn_log_path)
    unet_log = _load_loss_log(unet_log_path)

    required = ["train_l1", "val_l1", "train_perceptual", "val_perceptual"]
    if not all(key in srcnn_log for key in required) or not all(
        key in unet_log for key in required
    ):
        print("[visualize] Perceptual loss logs not found; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.86, wspace=0.25)

    series = [
        ("Total", "train_loss", "val_loss", "#1f77b4"),
        ("L1", "train_l1", "val_l1", "#ff7f0e"),
        ("Perceptual", "train_perceptual", "val_perceptual", "#2ca02c"),
    ]

    for ax, log, name in [
        (axes[0], srcnn_log, "SRCNN"),
        (axes[1], unet_log, "U-Net SR"),
    ]:
        epochs = range(1, len(log["train_loss"]) + 1)
        for label, train_key, val_key, color in series:
            ax.plot(epochs, log[train_key], label=f"{label} (train)",
                    color=color, linewidth=2, alpha=0.9)
            ax.plot(epochs, log[val_key], label=f"{label} (val)",
                    color=color, linewidth=2, linestyle="--", alpha=0.7)
        ax.set_title(f"{name} — Loss Components", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, ncol=2)

    fig.suptitle("Perceptual Loss Training Dynamics", fontsize=16, fontweight="bold")
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Perceptual loss curves saved → {save_path}")


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


def _resolve_paired_subject(subject_id: Optional[str] = None):
    """Return the paired 64mT and 3T file paths for one subject."""
    _, test_pairs = scan_bids_directory()

    if not test_pairs:
        raise RuntimeError("No paired 64mT/3T test subjects were found.")

    if subject_id is None:
        subject_id = cfg.PAIRED_SUBJECTS[0]

    for lr_path, hr_path in test_pairs:
        lr_subject = next((part for part in lr_path.parts if part.startswith("sub-")), None)
        hr_subject = next((part for part in hr_path.parts if part.startswith("sub-")), None)
        if lr_subject == subject_id and hr_subject == subject_id:
            return lr_path, hr_path

    raise ValueError(f"Could not find paired subject '{subject_id}'.")


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
        lr_np = all_lr[idx][0].numpy()
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

    col_titles = ["Bicubic |Error|", "SRCNN |Error|", "U-Net |Error|"]

    for row, idx in enumerate(indices):
        lr_t = all_lr[idx].unsqueeze(0).to(device)
        hr_np = all_hr[idx][0].numpy()
        lr_np = all_lr[idx][0].numpy()

        srcnn_pred = np.clip(srcnn(lr_t).cpu().numpy()[0, 0], 0, 1)
        unet_pred  = np.clip(unet(lr_t).cpu().numpy()[0, 0], 0, 1)

        err_srcnn = np.abs(hr_np - srcnn_pred)
        err_unet  = np.abs(hr_np - unet_pred)

        err_bicubic = np.abs(hr_np - lr_np)
        vmax_err = max(err_bicubic.max(), err_srcnn.max(), err_unet.max(), 0.01)

        def annotate_error(ax, err_map):
            threshold = np.percentile(err_map, 95)
            high_err = err_map > threshold
            if not high_err.any():
                return

            ys, xs = np.where(high_err)
            cy, cx = int(ys.mean()), int(xs.mean())
            h, w = err_map.shape

            annotations = []
            if cy < h * 0.35:
                annotations.append(("Cortical surface / Sulci", cx, cy))
            elif cy > h * 0.65:
                annotations.append(("Cerebellum / Brainstem", cx, cy))
            else:
                annotations.append(("WM/GM boundary", cx, cy))

            if len(ys) > 10:
                left_mask = xs < w // 2
                right_mask = xs >= w // 2
                if left_mask.any() and right_mask.any():
                    cy2 = int(ys[right_mask].mean())
                    cx2 = int(xs[right_mask].mean())
                    if abs(cy2 - cy) > 15 or abs(cx2 - cx) > 15:
                        annotations.append(("Ventricular boundary", cx2, cy2))

            for label, ax_x, ax_y in annotations[:2]:
                ax.annotate(
                    label,
                    xy=(ax_x, ax_y),
                    xytext=(ax_x + 20, ax_y - 25),
                    fontsize=9,
                    color="black",
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )

        # Panel 0: Bicubic error
        im0 = axes[row, 0].imshow(err_bicubic, cmap="coolwarm", vmin=0, vmax=vmax_err)
        axes[row, 0].set_title(col_titles[0] if row == 0 else "", fontsize=12,
                               fontweight="bold")
        axes[row, 0].axis("off")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        annotate_error(axes[row, 0], err_bicubic)

        # Panel 1: SRCNN error
        im1 = axes[row, 1].imshow(err_srcnn, cmap="coolwarm", vmin=0, vmax=vmax_err)
        axes[row, 1].set_title(col_titles[1] if row == 0 else "", fontsize=12,
                               fontweight="bold")
        axes[row, 1].axis("off")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        annotate_error(axes[row, 1], err_srcnn)

        # Panel 2: U-Net error
        im2 = axes[row, 2].imshow(err_unet, cmap="coolwarm", vmin=0, vmax=vmax_err)
        axes[row, 2].set_title(col_titles[2] if row == 0 else "", fontsize=12,
                               fontweight="bold")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        annotate_error(axes[row, 2], err_unet)

    plt.suptitle("Per-Pixel Error Maps — Synthetic Validation Set",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Error maps saved → {save_path}")


# ============================================================================
# 4. Research-ready triplet figure  (raw 64mT vs synthetic LR vs 3T)
# ============================================================================

def _select_slice_index(num_slices: int, slice_index: Optional[int] = None) -> int:
    """Choose a stable, central slice unless a specific index is supplied."""
    if num_slices <= 0:
        raise ValueError("Volume contains no usable slices.")

    if slice_index is None:
        return num_slices // 2

    return int(np.clip(slice_index, 0, num_slices - 1))


def _subject_name_from_path(path: Path) -> str:
    """Extract the first BIDS subject token from a file path."""
    return next(part for part in path.parts if part.startswith("sub-"))


def plot_triplet_comparison(subject_id: Optional[str] = None,
                            slice_index: Optional[int] = None,
                            save_path: Optional[str] = None):
    """Create a research-style triptych for one paired 64mT/3T subject.

    The left panel shows the raw 64mT T1w slice, the middle panel shows the
    corresponding synthetic 4× LR image (bicubic downsample + upsample), and
    the right panel shows the paired 3T reference slice.
    """
    if save_path is None:
        save_path = str(cfg.FIGURES_DIR / "triplet_comparison.png")

    lr_path, hr_path = _resolve_paired_subject(subject_id)
    subject_name = _subject_name_from_path(lr_path)

    lr_slices = extract_2d_slices(lr_path)
    hr_slices = extract_2d_slices(hr_path)
    n_slices = min(len(lr_slices), len(hr_slices))
    if n_slices == 0:
        raise RuntimeError("Paired subject has no overlapping usable slices.")

    idx = _select_slice_index(n_slices, slice_index)

    raw_lr = resize_slice(lr_slices[idx], cfg.IMAGE_SIZE)
    raw_hr = resize_slice(hr_slices[idx], cfg.IMAGE_SIZE)
    synthetic_lr = bicubic_upsample_np(
        bicubic_downsample(raw_lr, cfg.SCALE_FACTOR),
        raw_lr.shape,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.02, top=0.80, wspace=0.03)
    panel_titles = [
        f"(a) Raw 64 mT T1w slice\n{subject_name} · slice {idx}",
        f"(b) Synthetic 4× LR version\nBicubic downsample + upsample",
        f"(c) Paired 3T reference\n{subject_name} · slice {idx}",
    ]

    images = [raw_lr, synthetic_lr, raw_hr]
    cmap = "gray"
    norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, image, title in zip(axes, images, panel_titles):
        ax.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Normalized intensity", rotation=90, labelpad=12)

    fig.suptitle(
        "Cross-Scanner Slice Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Triplet comparison saved → {save_path}")


def plot_rician_noise_examples(subject_id: Optional[str] = None,
                               slice_index: Optional[int] = None,
                               save_path: Optional[str] = None,
                               sigma: Optional[float] = None):
    """Plot raw vs Rician-noised slice with a difference heatmap."""
    if save_path is None:
        save_path = str(cfg.FIGURES_DIR / "rician_noise_examples.png")
    if sigma is None:
        sigma = cfg.RICIAN_SIGMA

    lr_path, _ = _resolve_paired_subject(subject_id)
    subject_name = _subject_name_from_path(lr_path)

    lr_slices = extract_2d_slices(lr_path)
    if not lr_slices:
        raise RuntimeError("Selected subject has no usable 64mT slices.")

    idx = _select_slice_index(len(lr_slices), slice_index)
    raw_lr = resize_slice(lr_slices[idx], cfg.IMAGE_SIZE)

    raw_t = torch.from_numpy(raw_lr).unsqueeze(0).unsqueeze(0)
    noisy_lr = add_rician_noise(raw_t, sigma=sigma).squeeze(0).squeeze(0).numpy()
    diff = np.abs(noisy_lr - raw_lr)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.02, top=0.82, wspace=0.04)

    axes[0].imshow(raw_lr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title(f"Raw 64mT slice\n{subject_name} · slice {idx}",
                      fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(noisy_lr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Rician noise (sigma={sigma:.3f})", fontsize=11,
                      fontweight="bold", pad=8)
    axes[1].axis("off")

    diff_im = axes[2].imshow(diff, cmap="hot", vmin=0, vmax=max(diff.max(), 1e-3))
    axes[2].set_title("|Noisy - Raw|", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    cbar = fig.colorbar(diff_im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Absolute difference", rotation=90, labelpad=10)

    fig.suptitle("Rician Noise Augmentation", fontsize=16, fontweight="bold", y=0.95)
    plt.savefig(save_path, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Rician noise examples saved → {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    _, val_loader, _ = get_dataloaders()

    plot_loss_curves()
    plot_perceptual_loss_curves()
    plot_comparison_grid(val_loader, n_samples=5)
    plot_error_maps(val_loader, n_samples=3)
    plot_triplet_comparison()
    plot_rician_noise_examples()

    print("\n✓ All visualizations generated.")


if __name__ == "__main__":
    main()
