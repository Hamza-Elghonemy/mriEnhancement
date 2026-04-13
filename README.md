# IXI MRI Super-Resolution (Notebook Summary)

This repo contains `IXI_dataset.ipynb`, where a 2D MRI super-resolution pipeline was built and tested on IXI T1 scans.

## What Was Done (Brief)

- Loaded IXI `.nii/.nii.gz` files and handled corrupted/empty files safely.
- Built a preprocessing pipeline:
  - intensity normalization
  - 2D slice extraction from 3D MRI volumes (axial by default)
  - center crop/pad to fixed resolution
  - synthetic low-resolution generation (downsample + upsample)
- Split data into train/validation.
- Implemented and compared two SR models:
  - `SRCNN2D` (3-layer baseline)
  - `UNet2DSR` (advanced model)
- Added SR-specific losses/metrics:
  - Loss: `L1` / `MSE`
  - Metrics: `PSNR`, `SSIM`, plus per-sample difference maps.
- Added deterministic evaluation utilities and plotting:
  - fixed-slice comparisons (Before / After / GT)
  - fixed-set summary (improvement rates)
  - results dashboard graphs.
- Updated training setup:
  - residual learning (`SR = LR + model(LR)`)
  - deterministic validation set
  - longer training + LR scheduler.

## Dataset Properties

- Source modality: **IXI T1 MRI**
- File format: **NIfTI** (`.nii`, `.nii.gz`)
- Data nature: **3D brain volumes**
- Training setup in notebook: **2D SR on extracted slices**
- Validation setup: **deterministic center-slice evaluation** (to avoid random-slice bias)

## Results (Current Snapshot)

From a logged run in the notebook (before final residual/deterministic retraining pass):

- Baseline (LR->HR): `PSNR 25.903`, `SSIM 0.8059`
- SRCNN: `PSNR 26.615`, `SSIM 0.8444`
- UNet: `PSNR 27.442`, `SSIM 0.8783`

Also observed:

- Some individual slices were worse than baseline, so fixed-set evaluation was added to report improvement rate, not just single-sample visuals.

## Main File

- `IXI_dataset.ipynb`
