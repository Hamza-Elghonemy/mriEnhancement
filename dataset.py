"""
dataset.py — Data pipeline for MRI Super Resolution.

Handles BIDS NIfTI discovery, 3D→2D slice extraction, intensity
normalisation, synthetic LR/HR pair generation, and PyTorch Dataset /
DataLoader creation.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

import config as cfg


# ============================================================================
# 1. BIDS directory scanning
# ============================================================================

def scan_bids_directory() -> Tuple[List[Path], List[Tuple[Path, Path]]]:
    """Walk the BIDS tree and return file lists for training and testing.

    Returns
    -------
    train_files : list[Path]
        64mT T1w NIfTI files for subjects that do NOT appear in 3T
        (unpaired → used for synthetic LR/HR training).
    test_pairs : list[tuple[Path, Path]]
        (lr_64mt_path, hr_3t_path) for each paired test subject.
    """
    train_files: List[Path] = []
    test_pairs: List[Tuple[Path, Path]] = []

    paired_set = set(cfg.PAIRED_SUBJECTS)

    # --- Gather ALL 64mT T1w files (across sessions) ---
    for sub_dir in sorted(cfg.DIR_64MT.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        sub_name = sub_dir.name

        # Collect T1w files across all sessions
        t1w_files = sorted(
            sub_dir.rglob(f"*_{cfg.MODALITY}.nii.gz")
        )
        # Exclude localizer scans
        t1w_files = [f for f in t1w_files if "localizer" not in f.name.lower()]

        if sub_name in paired_set:
            # --- TEST subject: pair with 3T highres ---
            hr_path = cfg.DIR_3T / sub_name / "anat" / f"{sub_name}_acq-highres_{cfg.MODALITY}.nii.gz"
            if hr_path.exists() and t1w_files:
                # Use the first session's scan as LR input
                test_pairs.append((t1w_files[0], hr_path))
        else:
            # --- TRAIN subject: use for synthetic pairing ---
            train_files.extend(t1w_files)

    print(f"[dataset] Found {len(train_files)} training volumes, "
          f"{len(test_pairs)} paired test subjects.")
    return train_files, test_pairs


# ============================================================================
# 2. Volume loading & slice extraction
# ============================================================================

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """99th-percentile clipping followed by min-max scaling to [0, 1]."""
    p99 = np.percentile(vol, 99)
    vol = np.clip(vol, 0, p99)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 0:
        vol = (vol - vmin) / (vmax - vmin)
    return vol.astype(np.float32)


def extract_2d_slices(nifti_path: Path,
                      trim_frac: float = cfg.SLICE_TRIM_FRAC
                      ) -> List[np.ndarray]:
    """Load a 3D NIfTI volume and extract normalised 2D axial slices.

    Outer `trim_frac` % of slices are discarded from each end to remove
    noisy edge slices (FR-01.2).  Each returned slice is float32 in [0,1].
    """
    img = nib.load(str(nifti_path))
    vol = img.get_fdata().astype(np.float32)

    # Ensure 3D (some DWI may have 4th dim)
    if vol.ndim == 4:
        vol = vol[..., 0]

    vol = normalize_volume(vol)

    n_slices = vol.shape[2]  # axial = last dim for these BIDS volumes
    trim = max(1, int(n_slices * trim_frac))
    slices = []
    for i in range(trim, n_slices - trim):
        s = vol[:, :, i]
        # Skip near-empty slices (< 1 % foreground)
        if s.mean() < 0.01:
            continue
        slices.append(s)
    return slices


# ============================================================================
# 3. Synthetic LR/HR pair generation helpers
# ============================================================================

def bicubic_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsample 2D image by `factor` using bicubic interpolation."""
    h, w = img.shape
    small = resize(img, (h // factor, w // factor),
                   order=3, anti_aliasing=True, preserve_range=True)
    return small.astype(np.float32)


def bicubic_upsample_np(img: np.ndarray, target_shape: Tuple[int, int]
                        ) -> np.ndarray:
    """Upsample 2D image to `target_shape` using bicubic interpolation."""
    up = resize(img, target_shape, order=3,
                anti_aliasing=False, preserve_range=True)
    return up.astype(np.float32)


def resize_slice(img: np.ndarray, size: int = cfg.IMAGE_SIZE) -> np.ndarray:
    """Resize a 2D slice to (size, size)."""
    return resize(img, (size, size), order=3,
                  anti_aliasing=True, preserve_range=True).astype(np.float32)


# ============================================================================
# 4. PyTorch Datasets
# ============================================================================

class MRITrainDataset(Dataset):
    """Training dataset using synthetic LR/HR pairs from 64mT volumes.

    For each 64mT volume:
      HR = original 2D slice resized to IMAGE_SIZE × IMAGE_SIZE
      LR = bicubic ×4 downsample of HR, then bicubic upsample back to
           IMAGE_SIZE × IMAGE_SIZE  (SRCNN-style pre-upsampling)

    Random 64×64 patch cropping is applied for data augmentation (FR-01.5).
    """

    def __init__(self, nifti_paths: List[Path], patch_size: int = cfg.PATCH_SIZE,
                 augment: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        print(f"[MRITrainDataset] Loading slices from {len(nifti_paths)} volumes …")
        for path in nifti_paths:
            slices = extract_2d_slices(path)
            for s in slices:
                hr = resize_slice(s, cfg.IMAGE_SIZE)
                lr_small = bicubic_downsample(hr, cfg.SCALE_FACTOR)
                lr_up = bicubic_upsample_np(lr_small, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
                self.samples.append((lr_up, hr))

        print(f"[MRITrainDataset] Total training slices: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr, hr = self.samples[idx]

        # Random patch crop during training
        if self.augment and self.patch_size < cfg.IMAGE_SIZE:
            h, w = lr.shape
            top  = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            lr = lr[top:top + self.patch_size, left:left + self.patch_size]
            hr = hr[top:top + self.patch_size, left:left + self.patch_size]

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()

        # (H, W) → (1, H, W)
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        hr_t = torch.from_numpy(hr).unsqueeze(0)
        return lr_t, hr_t


class MRITestDataset(Dataset):
    """Test dataset using real paired 64mT (LR) / 3T-highres (HR) slices.

    Both LR and HR slices are resampled to IMAGE_SIZE × IMAGE_SIZE so
    that they are pixel-aligned for metric computation.
    """

    def __init__(self, test_pairs: List[Tuple[Path, Path]]):
        super().__init__()
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        print(f"[MRITestDataset] Loading slices from {len(test_pairs)} paired subjects …")
        for lr_path, hr_path in test_pairs:
            lr_slices = extract_2d_slices(lr_path)
            hr_slices = extract_2d_slices(hr_path)

            # Use the minimum slice count (volumes may differ in depth)
            n = min(len(lr_slices), len(hr_slices))
            for i in range(n):
                lr = resize_slice(lr_slices[i], cfg.IMAGE_SIZE)
                hr = resize_slice(hr_slices[i], cfg.IMAGE_SIZE)

                # Pre-upsample LR through down-then-up cycle (for SRCNN input)
                lr_small = bicubic_downsample(lr, cfg.SCALE_FACTOR)
                lr_up = bicubic_upsample_np(lr_small, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

                self.samples.append((lr_up, hr))

        print(f"[MRITestDataset] Total test slices: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr, hr = self.samples[idx]
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        hr_t = torch.from_numpy(hr).unsqueeze(0)
        return lr_t, hr_t


# ============================================================================
# 5. DataLoader factory
# ============================================================================

def get_dataloaders(
    batch_size: int = cfg.BATCH_SIZE,
    num_workers: int = cfg.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Train/val split is done at the *subject* level (not slice level) to
    prevent data leakage.  A fixed random seed ensures reproducibility.
    """
    train_files, test_pairs = scan_bids_directory()

    # --- Subject-level train / val split ---
    rng = random.Random(cfg.SEED)
    rng.shuffle(train_files)
    n_val = max(1, int(len(train_files) * cfg.VAL_RATIO))
    val_files  = train_files[:n_val]
    trn_files  = train_files[n_val:]

    print(f"[dataloaders] Train volumes: {len(trn_files)}, "
          f"Val volumes: {len(val_files)}, "
          f"Test pairs: {len(test_pairs)}")

    train_ds = MRITrainDataset(trn_files, patch_size=cfg.PATCH_SIZE, augment=True)
    val_ds   = MRITrainDataset(val_files, patch_size=cfg.IMAGE_SIZE, augment=False)
    test_ds  = MRITestDataset(test_pairs)

    pin = cfg.DEVICE.type == "cuda"  # pin_memory only useful for CUDA
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


# ============================================================================
# Quick sanity check
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")

    # Peek at one batch
    lr, hr = next(iter(train_loader))
    print(f"Train batch — LR: {lr.shape}, HR: {hr.shape}, "
          f"LR range: [{lr.min():.3f}, {lr.max():.3f}], "
          f"HR range: [{hr.min():.3f}, {hr.max():.3f}]")
