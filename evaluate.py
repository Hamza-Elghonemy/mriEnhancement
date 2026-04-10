"""
evaluate.py — Evaluation pipeline for MRI Super Resolution.

Two evaluation modes:
  A. Synthetic evaluation (primary) — val set with bicubic-degraded LR/HR pairs
     from 64mT volumes.  Same domain as training → meaningful PSNR/SSIM.
  B. Cross-scanner evaluation (secondary) — real 64mT vs 3T paired test set.
     Different domains → expected low PSNR. Reported for reference only.

Prints formatted comparison tables and saves results to JSON.
"""

import json
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import (
    peak_signal_noise_ratio as compute_psnr,
    structural_similarity as compute_ssim,
)

import config as cfg
from dataset import get_dataloaders
from models import SRCNN, UNetSR


# ============================================================================
# Metric helpers
# ============================================================================

def compute_metrics(pred: np.ndarray, target: np.ndarray
                    ) -> Tuple[float, float]:
    """Compute PSNR (dB) and SSIM between two 2D images.

    Both must be float in [0, 1] with data_range=1.0.
    """
    psnr = compute_psnr(target, pred, data_range=1.0)
    ssim = compute_ssim(target, pred, data_range=1.0)
    return psnr, ssim


# ============================================================================
# Per-method evaluation
# ============================================================================

@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device = cfg.DEVICE
                   ) -> Tuple[List[float], List[float]]:
    """Run model inference on a DataLoader, return per-slice PSNR & SSIM."""
    model.eval()
    model.to(device)
    psnrs, ssims = [], []

    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        pred = model(lr_batch).cpu().numpy()

        for i in range(pred.shape[0]):
            p = np.clip(pred[i, 0], 0, 1)
            h = hr_batch[i, 0].numpy()
            psnr, ssim = compute_metrics(p, h)
            psnrs.append(psnr)
            ssims.append(ssim)

    return psnrs, ssims


def evaluate_bicubic(loader: DataLoader
                     ) -> Tuple[List[float], List[float]]:
    """Evaluate the bicubic baseline (input already contains bicubic-upsampled LR)."""
    psnrs, ssims = [], []

    for lr_batch, hr_batch in loader:
        for i in range(lr_batch.shape[0]):
            lr_np = lr_batch[i, 0].numpy()
            hr_np = hr_batch[i, 0].numpy()
            psnr, ssim = compute_metrics(lr_np, hr_np)
            psnrs.append(psnr)
            ssims.append(ssim)

    return psnrs, ssims


# ============================================================================
# Results table
# ============================================================================

def print_results_table(results: Dict[str, Dict[str, List[float]]],
                        title: str = "Evaluation Results",
                        save_name: str = "evaluation_results.json"):
    """Print a formatted comparison table and save to JSON."""

    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)
    print(f"  {'Method':<20} {'PSNR (dB)':<22} {'SSIM':<22}")
    print("-" * 65)

    summary = {}
    for method, metrics in results.items():
        psnr_arr = np.array(metrics["psnr"])
        ssim_arr = np.array(metrics["ssim"])
        psnr_str = f"{psnr_arr.mean():.2f} ± {psnr_arr.std():.2f}"
        ssim_str = f"{ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}"
        print(f"  {method:<20} {psnr_str:<22} {ssim_str:<22}")
        summary[method] = {
            "psnr_mean": float(psnr_arr.mean()),
            "psnr_std":  float(psnr_arr.std()),
            "ssim_mean": float(ssim_arr.mean()),
            "ssim_std":  float(ssim_arr.std()),
            "n_slices":  len(psnr_arr),
        }

    print("=" * 65)

    out_path = cfg.OUTPUT_DIR / save_name
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved to {out_path}")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    train_loader, val_loader, test_loader = get_dataloaders()

    # Load trained models
    srcnn = SRCNN()
    srcnn.load_state_dict(torch.load(str(cfg.SRCNN_CKPT), map_location=cfg.DEVICE,
                                     weights_only=True))

    unet = UNetSR()
    unet.load_state_dict(torch.load(str(cfg.UNET_CKPT), map_location=cfg.DEVICE,
                                    weights_only=True))

    # ==================================================================
    # A. SYNTHETIC EVALUATION (primary — same domain as training)
    #    val_loader has synthetic LR/HR pairs from held-out 64mT volumes.
    #    LR = bicubic ×4 downsample-then-upsample of the original slice.
    #    HR = original slice resized to 256×256.
    # ==================================================================
    print("\n" + "▸" * 50)
    print("  SYNTHETIC EVALUATION (validation set)")
    print("  LR = bicubic-degraded 64mT  |  HR = original 64mT")
    print("▸" * 50)

    syn_results: Dict[str, Dict[str, List[float]]] = {}

    print("\n  Evaluating: Bicubic Interpolation (synthetic) …")
    psnrs, ssims = evaluate_bicubic(val_loader)
    syn_results["Bicubic"] = {"psnr": psnrs, "ssim": ssims}
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")

    print("  Evaluating: SRCNN (synthetic) …")
    psnrs, ssims = evaluate_model(srcnn, val_loader)
    syn_results["SRCNN"] = {"psnr": psnrs, "ssim": ssims}
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")

    print("  Evaluating: U-Net SR (synthetic) …")
    psnrs, ssims = evaluate_model(unet, val_loader)
    syn_results["U-Net SR"] = {"psnr": psnrs, "ssim": ssims}
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")

    print_results_table(syn_results,
                        title="Synthetic Evaluation (same domain — primary)",
                        save_name="evaluation_results.json")

    # ==================================================================
    # B. CROSS-SCANNER EVALUATION (reference — different domains)
    #    test_loader has real 64mT (LR) vs 3T-highres (HR) pairs.
    #    Low PSNR expected due to contrast/geometry mismatch.
    # ==================================================================
    print("\n" + "▸" * 50)
    print("  CROSS-SCANNER EVALUATION (test set)")
    print("  LR = real 64mT  |  HR = real 3T highres")
    print("  ⚠ Different scanners, no registration → low PSNR expected")
    print("▸" * 50)

    cross_results: Dict[str, Dict[str, List[float]]] = {}

    print("\n  Evaluating: Bicubic (cross-scanner) …")
    psnrs, ssims = evaluate_bicubic(test_loader)
    cross_results["Bicubic"] = {"psnr": psnrs, "ssim": ssims}
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")

    print("  Evaluating: SRCNN (cross-scanner) …")
    psnrs, ssims = evaluate_model(srcnn, test_loader)
    cross_results["SRCNN"] = {"psnr": psnrs, "ssim": ssims}
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")

    print("  Evaluating: U-Net SR (cross-scanner) …")
    psnrs, ssims = evaluate_model(unet, test_loader)
    cross_results["U-Net SR"] = {"psnr": psnrs, "ssim": ssims}
    print(f"    PSNR: {np.mean(psnrs):.2f}  SSIM: {np.mean(ssims):.4f}")

    print_results_table(cross_results,
                        title="Cross-Scanner Evaluation (64mT→3T — reference)",
                        save_name="evaluation_cross_scanner.json")


if __name__ == "__main__":
    main()
