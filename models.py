"""
models.py — Model architectures for MRI Super Resolution.

Contains:
  1. bicubic_upsample()       – non-learnable baseline  (FR-02)
  2. SRCNN                    – 3-layer CNN              (FR-03)
  3. UNetSR                   – 4-stage U-Net            (FR-04)
  4. PerceptualLoss           – VGG-19 feature loss      (BR-01, bonus)
  5. add_rician_noise()       – Rician noise injection    (BR-02, bonus)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize as sk_resize


# ============================================================================
# 1. Bicubic baseline (non-learnable)
# ============================================================================

def bicubic_upsample(lr_image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bicubic interpolation baseline using skimage (order=3).

    Parameters
    ----------
    lr_image : ndarray, shape (H_lr, W_lr)
        Low-resolution input (2D, float32, [0,1]).
    target_shape : tuple (H_hr, W_hr)
        Desired output spatial size.

    Returns
    -------
    ndarray, shape target_shape, float32 in [0,1].
    """
    return sk_resize(lr_image, target_shape, order=3,
                     anti_aliasing=False, preserve_range=True
                     ).astype(np.float32)


# ============================================================================
# 2. SRCNN  (FR-03)
# ============================================================================

class SRCNN(nn.Module):
    """Super-Resolution CNN — 3 convolutional layers.

    Architecture (per Dong et al. 2014, adapted for MRI):
        Conv(1→64, 9×9) → ReLU
        Conv(64→32, 1×1) → ReLU
        Conv(32→1,  5×5)

    Input must be the bicubic-upsampled LR image (same spatial size as HR).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32,  1, kernel_size=5, padding=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# ============================================================================
# 3. U-Net SR  (FR-04)
# ============================================================================

class _ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSR(nn.Module):
    """U-Net for Super Resolution with 4 encoder stages + bottleneck + 4 decoder stages.

    Features: [64, 128, 256, 512] → bottleneck 1024 → decoder mirrors encoder.
    Skip connections concatenate encoder features with decoder features.
    Transposed convolutions are used for upsampling (FR-04.3).
    """

    def __init__(self):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = _ConvBlock(1, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.enc4 = _ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = _ConvBlock(512, 1024)

        # ---- Decoder (transposed conv for upsampling) ----
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _ConvBlock(1024, 512)   # 512 (up) + 512 (skip)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(512, 256)    # 256 + 256

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(256, 128)    # 128 + 128

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(128, 64)     # 64 + 64

        # ---- Final 1×1 conv to single channel ----
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = self._pad_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.final(d1)

    @staticmethod
    def _pad_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle spatial size mismatch between decoder output and skip."""
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([x, skip], dim=1)


# ============================================================================
# 4. Perceptual Loss (Bonus BR-01)
# ============================================================================

class PerceptualLoss(nn.Module):
    """Feature-level loss using frozen VGG-19 up to relu2_2.

    Grayscale (1-channel) inputs are repeated to 3 channels before
    being passed through VGG.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features

        # relu2_2 is at index 8 in vgg19.features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Convert grayscale to 3-ch and apply ImageNet normalisation."""
        x = x.repeat(1, 3, 1, 1)   # (B, 1, H, W) → (B, 3, H, W)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat   = self.feature_extractor(self._prepare(pred))
        target_feat = self.feature_extractor(self._prepare(target))
        return F.l1_loss(pred_feat, target_feat)


# ============================================================================
# 5. Rician noise injection (Bonus BR-02)
# ============================================================================

def add_rician_noise(image: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """Add Rician-distributed noise to an image tensor.

    Rician noise model:  sqrt((x + n1)^2 + n2^2)
    where n1, n2 ~ N(0, sigma^2).

    Parameters
    ----------
    image : Tensor, shape (B, 1, H, W) or (1, H, W)
    sigma : float, noise level

    Returns
    -------
    Noisy image tensor (same shape), clipped to [0, 1].
    """
    n1 = torch.randn_like(image) * sigma
    n2 = torch.randn_like(image) * sigma
    noisy = torch.sqrt((image + n1) ** 2 + n2 ** 2)
    return torch.clamp(noisy, 0.0, 1.0)


# ============================================================================
# Quick shape verification
# ============================================================================
if __name__ == "__main__":
    # Verify forward pass shapes
    dummy = torch.randn(2, 1, 64, 64)

    srcnn = SRCNN()
    out_s = srcnn(dummy)
    print(f"SRCNN:  in={dummy.shape} → out={out_s.shape}")

    dummy256 = torch.randn(2, 1, 256, 256)
    unet = UNetSR()
    out_u = unet(dummy256)
    print(f"U-Net:  in={dummy256.shape} → out={out_u.shape}")

    # Perceptual loss
    ploss = PerceptualLoss()
    l = ploss(dummy256, dummy256)
    print(f"Perceptual loss: {l.item():.6f}")

    # Rician noise
    noisy = add_rician_noise(dummy256, sigma=0.05)
    print(f"Rician noise: range [{noisy.min():.3f}, {noisy.max():.3f}]")

    print("\n✓ All model shapes verified.")
