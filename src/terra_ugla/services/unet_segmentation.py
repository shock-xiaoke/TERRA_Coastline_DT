"""UNet-based waterline segmentation utilities with safe runtime fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rasterio.transform import xy
from shapely.geometry import LineString

from .intersections import transform_geometry

MODEL_INPUT_SIZE = (512, 512)  # width, height

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
UNET_CHECKPOINT_PATH = _PACKAGE_ROOT / "models" / "ml_models" / "robust_unet_best.pth"
LEGACY_UNET_CHECKPOINT_PATH = Path("segmentation_and_prediction") / "segment_with_Unet" / "robust_unet_best.pth"
UNET_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
UNET_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class UnetSegmentationResult:
    waterline_src: LineString | None
    waterline_wgs84: LineString | None
    waterline_utm: LineString | None
    threshold: float
    method: str
    model_path: str | None
    error: str | None


def resolve_unet_checkpoint_path() -> Path:
    env_path = os.getenv("TERRA_UNET_MODEL_PATH")
    if env_path:
        return Path(env_path)
    if UNET_CHECKPOINT_PATH.exists():
        return UNET_CHECKPOINT_PATH
    return LEGACY_UNET_CHECKPOINT_PATH


def _try_import_torch() -> tuple[Any | None, Any | None, Any | None]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception:
        return None, None, None


def _build_model(torch, nn):
    class ChannelAttention(nn.Module):
        def __init__(self, in_channels: int, ratio: int = 16):
            super().__init__()
            mid = max(1, in_channels // ratio)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(nn.Conv2d(in_channels, mid, 1, bias=False), nn.ReLU(), nn.Conv2d(mid, in_channels, 1, bias=False))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            return x * self.sigmoid(avg_out + max_out)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size: int = 7):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = x.mean(dim=1, keepdim=True)
            max_out, _ = x.max(dim=1, keepdim=True)
            x_att = self.conv1(torch.cat([avg_out, max_out], dim=1))
            return x * self.sigmoid(x_att)

    class AttentionGate(nn.Module):
        def __init__(self, f_g: int, f_l: int, f_int: int):
            super().__init__()
            self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, bias=True), nn.BatchNorm2d(f_int))
            self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, bias=True), nn.BatchNorm2d(f_int))
            self.psi = nn.Sequential(nn.Conv2d(f_int, 1, kernel_size=1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            psi = self.relu(self.w_g(g) + self.w_x(x))
            return x * self.psi(psi)

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(dropout_rate)
            self.relu = nn.ReLU(inplace=True)
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = self.sa(self.ca(out))
            return self.relu(out + residual)

    class DilatedBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            q = out_channels // 4
            self.conv1 = nn.Conv2d(in_channels, q, 1)
            self.conv2 = nn.Conv2d(in_channels, q, 3, padding=1, dilation=1)
            self.conv3 = nn.Conv2d(in_channels, q, 3, padding=2, dilation=2)
            self.conv4 = nn.Conv2d(in_channels, q, 3, padding=4, dilation=4)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)))

    class RobustUNet(nn.Module):
        def __init__(self, n_channels: int = 3, n_classes: int = 1, base_channels: int = 64):
            super().__init__()
            self.inc = ResidualBlock(n_channels, base_channels, dropout_rate=0.1)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels, base_channels * 2, dropout_rate=0.1))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 2, base_channels * 4, dropout_rate=0.2))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(base_channels * 4, base_channels * 8, dropout_rate=0.2))
            self.bottleneck = nn.Sequential(
                nn.MaxPool2d(2),
                DilatedBlock(base_channels * 8, base_channels * 16),
                ResidualBlock(base_channels * 16, base_channels * 16, dropout_rate=0.3),
            )

            self.att4 = AttentionGate(base_channels * 8, base_channels * 8, base_channels * 4)
            self.att3 = AttentionGate(base_channels * 4, base_channels * 4, base_channels * 2)
            self.att2 = AttentionGate(base_channels * 2, base_channels * 2, base_channels)
            self.att1 = AttentionGate(base_channels, base_channels, max(1, base_channels // 2))

            self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
            self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8, dropout_rate=0.2)
            self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
            self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4, dropout_rate=0.2)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2, dropout_rate=0.1)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = ResidualBlock(base_channels * 2, base_channels, dropout_rate=0.1)
            self.outc = nn.Sequential(nn.Conv2d(base_channels, n_classes, 1), nn.Sigmoid())

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.bottleneck(x4)
            x = self.up4(x5)
            x = torch.cat([self.att4(x, x4), x], dim=1)
            x = self.dec4(x)
            x = self.up3(x)
            x = torch.cat([self.att3(x, x3), x], dim=1)
            x = self.dec3(x)
            x = self.up2(x)
            x = torch.cat([self.att2(x, x2), x], dim=1)
            x = self.dec2(x)
            x = self.up1(x)
            x = torch.cat([self.att1(x, x1), x], dim=1)
            x = self.dec1(x)
            return self.outc(x)

    return RobustUNet()


@lru_cache(maxsize=1)
def _load_unet_model(checkpoint_path: str) -> tuple[Any | None, Any | None, str | None]:
    torch, nn, _ = _try_import_torch()
    if torch is None or nn is None:
        return None, None, "PyTorch is not installed; UNet segmentation disabled."

    path = Path(checkpoint_path)
    if not path.exists():
        return None, None, f"UNet checkpoint not found: {path}"

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(torch, nn).to(device)
        loaded = torch.load(path, map_location=device)
        state_dict = loaded
        if isinstance(loaded, dict):
            state_dict = (
                loaded.get("state_dict")
                or loaded.get("model_state_dict")
                or loaded.get("model")
                or loaded
            )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, device, None
    except Exception as exc:
        return None, None, f"Failed to load UNet checkpoint: {exc}"


def _prepare_rgb(im_ms: np.ndarray) -> np.ndarray:
    # UnetV training pipeline uses RGB + ImageNet normalization.
    # Input stack is [B, G, R, ...], so reorder to [R, G, B].
    rgb = np.stack([im_ms[:, :, 2], im_ms[:, :, 1], im_ms[:, :, 0]], axis=-1).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-channel robust scaling into [0, 1] before normalization.
    for channel in range(3):
        band = rgb[:, :, channel]
        p2 = float(np.nanpercentile(band, 2.0))
        p98 = float(np.nanpercentile(band, 98.0))
        if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
            p2, p98 = float(np.nanmin(band)), float(np.nanmax(band))
        if np.isfinite(p2) and np.isfinite(p98) and p98 > p2:
            rgb[:, :, channel] = (band - p2) / (p98 - p2)
        else:
            rgb[:, :, channel] = 0.0

    rgb = np.clip(rgb, 0.0, 1.0).transpose(2, 0, 1)
    rgb = (rgb - UNET_RGB_MEAN) / UNET_RGB_STD
    return rgb.astype(np.float32)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except Exception:
        return mask

    labeled, count = ndimage.label(mask.astype(np.uint8))
    if count <= 1:
        return mask

    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    if sizes is None or len(sizes) == 0:
        return mask
    largest_label = int(np.argmax(sizes) + 1)
    return labeled == largest_label


def _median_smooth(y: np.ndarray, k: int = 9) -> np.ndarray:
    if y.size < k or k < 3 or k % 2 == 0:
        return y
    radius = k // 2
    ys = y.copy()
    for idx in range(radius, y.size - radius):
        ys[idx] = float(np.median(y[idx - radius : idx + radius + 1]))
    return ys


def _extract_lower_boundary(mask: np.ndarray, x_step: int = 1, min_run: int = 12) -> np.ndarray | None:
    h, w = mask.shape
    xs = np.arange(0, w, x_step, dtype=np.int32)
    y_bottom = np.full(xs.shape, -1, dtype=np.int32)

    for idx, x in enumerate(xs):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size > 0:
            y_bottom[idx] = int(ys.max())

    valid = y_bottom >= 0
    xs = xs[valid]
    y_bottom = y_bottom[valid]
    if xs.size < 4:
        return None

    gaps = np.where(np.diff(xs) > x_step)[0]
    segments = np.split(np.arange(xs.size), gaps + 1)
    kept = [seg for seg in segments if seg.size >= min_run]
    keep_idx = np.concatenate(kept) if kept else np.arange(xs.size)

    xs = xs[keep_idx]
    y_bottom = y_bottom[keep_idx]
    y_bottom = _median_smooth(y_bottom.astype(np.float32), k=9)
    coords = np.column_stack([xs.astype(np.float32), y_bottom.astype(np.float32)])

    if len(coords) < 4:
        return None
    return coords


def _coords_to_linestring(pixel_coords: np.ndarray, affine_transform) -> LineString | None:
    if pixel_coords is None or len(pixel_coords) < 2:
        return None

    cols = pixel_coords[:, 0]
    rows = pixel_coords[:, 1]
    xs, ys = xy(affine_transform, rows.tolist(), cols.tolist(), offset="center")
    line = LineString(list(zip(xs, ys)))
    if line.is_empty or line.length <= 0:
        return None
    return line


def segment_waterline_from_multiband(
    im_ms: np.ndarray,
    affine_transform,
    src_epsg: int,
    utm_epsg: int,
    cloud_mask: np.ndarray | None = None,
    aoi_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> UnetSegmentationResult:
    """Predict a water mask via robust UNet and convert it to a coastline line."""
    torch, _, F = _try_import_torch()
    checkpoint_path = resolve_unet_checkpoint_path()
    model, device, load_error = _load_unet_model(str(checkpoint_path))
    if load_error or model is None or torch is None or F is None or device is None:
        return UnetSegmentationResult(
            waterline_src=None,
            waterline_wgs84=None,
            waterline_utm=None,
            threshold=float(threshold),
            method="coastguard_fallback",
            model_path=str(checkpoint_path),
            error=load_error or "UNet runtime unavailable.",
        )

    try:
        height, width = im_ms.shape[:2]
        rgb = _prepare_rgb(im_ms)
        inp = torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32)
        inp_resized = F.interpolate(inp, size=(MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]), mode="bilinear", align_corners=False)

        with torch.no_grad():
            pred = model(inp_resized).detach().cpu().numpy()[0, 0]

        pred_mask = pred >= float(threshold)
        mask_img = Image.fromarray((pred_mask.astype(np.uint8) * 255))
        mask_img = mask_img.resize((width, height), Image.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8) > 0

        if cloud_mask is not None:
            mask = np.logical_and(mask, ~cloud_mask)
        if aoi_mask is not None:
            mask = np.logical_and(mask, aoi_mask)
        mask = _keep_largest_component(mask)

        pixels = _extract_lower_boundary(mask.astype(np.uint8))
        line_src = _coords_to_linestring(pixels, affine_transform)
        if line_src is None:
            return UnetSegmentationResult(
                waterline_src=None,
                waterline_wgs84=None,
                waterline_utm=None,
                threshold=float(threshold),
                method="coastguard_fallback",
                model_path=str(checkpoint_path),
                error="UNet mask produced no usable coastline boundary.",
            )

        line_wgs84 = line_src if src_epsg == 4326 else transform_geometry(line_src, src_epsg, 4326)
        line_utm = line_src if src_epsg == utm_epsg else transform_geometry(line_src, src_epsg, utm_epsg)

        return UnetSegmentationResult(
            waterline_src=line_src,
            waterline_wgs84=line_wgs84,
            waterline_utm=line_utm,
            threshold=float(threshold),
            method="unet_robust",
            model_path=str(checkpoint_path),
            error=None,
        )
    except Exception as exc:
        return UnetSegmentationResult(
            waterline_src=None,
            waterline_wgs84=None,
            waterline_utm=None,
            threshold=float(threshold),
            method="coastguard_fallback",
            model_path=str(checkpoint_path),
            error=f"UNet inference failed: {exc}",
        )
