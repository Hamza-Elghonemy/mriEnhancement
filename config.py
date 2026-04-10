"""
config.py — Central configuration for MRI Super Resolution project.

All hyperparameters, paths, and constants are defined here so that
every other module imports from a single source of truth.
"""

import os
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Device selection (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Raw BIDS dataset delivered by Zenodo
DATA_ROOT = PROJECT_ROOT / (
    "Paired 64mT and 3T Brain MRI Scans of Healthy Subjects "
    "for Neuroimaging Research"
) / "Data"
DIR_3T  = DATA_ROOT / "3T data"
DIR_64MT = DATA_ROOT / "64mT data"

# Outputs
OUTPUT_DIR      = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR  = OUTPUT_DIR / "checkpoints"
FIGURES_DIR     = OUTPUT_DIR / "figures"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint file names
SRCNN_CKPT = CHECKPOINT_DIR / "srcnn_mri.pth"
UNET_CKPT  = CHECKPOINT_DIR / "unet_sr_mri.pth"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
# Subjects present in BOTH 3T and 64mT directories (true paired test set)
PAIRED_SUBJECTS = [
    "sub-0011", "sub-0015", "sub-0023", "sub-0025", "sub-0027",
    "sub-0035", "sub-0046", "sub-0047", "sub-0048", "sub-0064",
]

# Modality to use (start with T1-weighted only)
MODALITY = "T1w"

# Target 2D slice size after resampling (both LR-upsampled and HR)
IMAGE_SIZE = 256

# Fraction of outer slices to skip from each end of the volume (noisy edges)
SLICE_TRIM_FRAC = 0.10

# Scale factor for synthetic bicubic downsampling during training
SCALE_FACTOR = 4

# Random patch size for training augmentation
PATCH_SIZE = 64

# Train / val split ratio (applied to train-only 64mT subjects)
VAL_RATIO = 0.20

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
BATCH_SIZE    = 16
NUM_EPOCHS    = 50
NUM_WORKERS   = 2

# StepLR scheduler (mitigates overfitting on small dataset)
LR_STEP_SIZE  = 20
LR_GAMMA      = 0.5
