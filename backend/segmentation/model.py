"""
3D Volumetric Brain Tumor Segmentation Model
Native PyTorch 3D U-Net architecture designed for multi-compartment BraTS segmentation.
Supports:
  - WT (Whole Tumor): Labels 1 + 2 + 3 (Necrotic Core + Vasogenic Edema + Enhancing Margin)
  - TC (Tumor Core): Labels 1 + 3 (Necrotic Core + Enhancing Margin)
  - ET (Enhancing Tumor): Label 3 (Active Contrast-Enhancing Neoplasm)
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# PyTorch is optional — only needed for local training.
# On Render (requirements-render.txt) torch is omitted to keep image lean.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None   # type: ignore
    nn = None      # type: ignore


LABEL_DEFINITIONS = {
    0: {"name": "Background", "tag": "BG", "color": "transparent"},
    1: {"name": "Necrotic / Non-Enhancing Core", "tag": "NCR/NET", "color": "#FF3B30"},
    2: {"name": "Peritumoral Vasogenic Edema", "tag": "ED", "color": "#39D5FF"},
    3: {"name": "Active Contrast-Enhancing Margin", "tag": "ET", "color": "#FFD60A"}
}

SUBREGION_HIERARCHY = {
    "WT": {
        "name": "Whole Tumor",
        "labels": [1, 2, 3],
        "color": "#39D5FF",
        "desc": "Entire macroscopic neoplastic mass (Necrotic Core + Edema + Enhancing Margin)"
    },
    "TC": {
        "name": "Tumor Core",
        "labels": [1, 3],
        "color": "#FF3B30",
        "desc": "Aggressive tumor core resected during cytoreductive surgery"
    },
    "ET": {
        "name": "Enhancing Tumor",
        "labels": [3],
        "color": "#FFD60A",
        "desc": "Hypervascular, breakdown-of-BBB active viable tumor margin"
    }
}


if TORCH_AVAILABLE:
    class DoubleConv3D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(0.01, inplace=True)
            )
            self.residual = (
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
                if in_channels != out_channels else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class NativeBrainTumorUNet3D(nn.Module):
    """
    Volumetric 3D U-Net with skip connections and Instance Normalization.
    Takes 3D multi-parametric NIfTI volumes and predicts 4-class voxel segmentation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv3D(in_channels, channels[0])
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(channels[0], channels[1]))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(channels[1], channels[2]))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(channels[2], channels[3]))
        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Dropout3d(dropout_prob),
            DoubleConv3D(channels[3], channels[4])
        )

        # Decoder
        self.up1 = nn.ConvTranspose3d(channels[4], channels[3], kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(channels[4], channels[3])

        self.up2 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(channels[3], channels[2])

        self.up3 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(channels[2], channels[1])

        self.up4 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv3D(channels[1], channels[0])

        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))

        return self.outc(x)


def load_segmentation_model(
    model_path: Optional[str] = None,
    device=None
):
    """Load 3D U-Net weights. Returns (model, is_loaded, meta).
    Returns (None, False, meta) when PyTorch is not installed."""
    meta = {
        "device": "cpu",
        "architecture": "NativeBrainTumorUNet3D (BraTS Volumetric Benchmark)",
        "in_channels": 1,
        "out_channels": 4,
        "checkpoint": None
    }

    if not TORCH_AVAILABLE:
        print("[INFO] PyTorch not installed — 3D U-Net weights not loaded (numpy pipeline active)")
        return None, False, meta

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta["device"] = str(device)

    if model_path:
        p = Path(model_path)
        if p.exists():
            try:
                ckpt = torch.load(str(p), map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                is_loaded = True
                meta["checkpoint"] = p.name
                print(f"[INFO] Loaded 3D Segmentation checkpoint: {p.name}")
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint weights from {p}: {e}")

    model.eval()
    return model, is_loaded, meta
