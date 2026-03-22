#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal VE forecasting — year-ahead, irregularity-aware, MC Dropout uncertainty.

== Product goal ==
  A user selects an AOI and a forecast horizon x ∈ {1,2,3,4,5} years.
  The model returns the predicted VE probability map at x years ahead,
  together with a spatial uncertainty heatmap from MC Dropout.

== Key design principles ==
  1. Year-conditioned forecasting (1-5 yr horizons; internally stored as months).
  2. Irregular time series — no interpolation of missing months.
     Each training sample pairs a window of actual observations with a real
     future observation matched by calendar time (±tolerance).
  3. Explicit per-observation temporal conditioning: every history token carries
     a sinusoidal embedding of its real time-offset from the reference date,
     so the model knows observation spacing, not just ordering.
  4. Padded history slots carry a sentinel offset (_PAD_OFFSET_MONTHS) and are
     explicitly zeroed in the temporal module via a boolean valid_mask [B, T].
  5. Evaluation split is strictly by AOI if --val-aois is given; otherwise a
     temporal split is used (last val_frac of calendar time as validation).
  6. MC Dropout uncertainty: N stochastic forward passes →
       central estimate (mean), uncertainty map (std), 80% envelope.
  7. Inference outputs include a GeoJSON of the predicted VE line +
     confidence envelopes, and a structured forecast_metadata.json.

== Three modes ==
  --mode prepare   Render LabelMe JSON VE polylines to cached binary-mask NPY files.
  --mode train     Train VETemporalForecaster from cached masks.
  --mode infer     Probabilistic year-ahead forecast with uncertainty outputs.

== Architecture ==
  SpatialEncoder       : [1, H, W]  →  [latent_dim]  (per-frame CNN, weight-shared)
  SinusoidalEmbedding  : scalar t  →  [time_emb_dim]  (freqs cached in __init__)
  HorizonEncoder       : scalar yrs →  [h_dim]  (sinusoidal + MLP)
  TemporalModule       : [B,T,latent+time+h] → Mamba-Lite × N → LSTM → [latent]
                         (padded tokens zeroed via valid_mask before and after Mamba)
  SpatialDecoder       : [latent + h_dim] → [1, H, W]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        distance_transform_edt,
        label as nd_label,
    )
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Sentinel offset value placed into padded (non-observed) history slots.
# Chosen to be outside the realistic data range (~120 months of history)
# so the model can learn to down-weight these positions.
_PAD_OFFSET_MONTHS: float = 999.0

# Regex for extracting (aoi, year, month) from a LabelMe JSON filename.
# Handles patterns like: aoi_01_202101__S2A_MSIL2A_...SAFE.json
_AOI_DATE_RE = re.compile(r"(aoi_\d+(?:_holdout)?)_(20\d{2})(\d{2})", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 0.  CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VE temporal forecaster: prepare / train / infer modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["prepare", "train", "infer"], default="train")

    # ── data paths ──────────────────────────────────────────────────────────
    p.add_argument("--labelme-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "labelme_work")
    p.add_argument("--masks-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "ve_masks",
                   help="Where cached binary-mask NPY files are read/written.")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "models" / "ve_forecaster")

    # ── mask rendering ───────────────────────────────────────────────────────
    p.add_argument("--mask-size", type=int, default=128,
                   help="Square spatial resolution for the forecaster's VE masks.")
    p.add_argument("--line-width", type=int, default=4,
                   help="Line width (px at native resolution) for VE polyline rendering.")

    # ── dataset ─────────────────────────────────────────────────────────────
    p.add_argument("--history-len", type=int, default=12,
                   help="Number of historical observation slots (padded if data is sparse).")
    p.add_argument("--min-history", type=int, default=4,
                   help="Minimum real (non-padded) observations required to build a sample.")
    p.add_argument("--horizon-years", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                   help="Training forecast horizons in years.")
    p.add_argument("--target-tolerance", type=int, default=2,
                   help="Months of tolerance when matching a future observation to the target date.")
    p.add_argument("--val-aois", nargs="*", default=None,
                   help="AOIs reserved for validation (AOI-split mode). "
                        "If omitted, a temporal split is used instead.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of calendar time reserved for validation "
                        "when --val-aois is not specified (temporal-split mode).")
    p.add_argument("--holdout-aois", nargs="*", default=[],
                   help="AOIs held out entirely from training and validation.")

    # ── model ────────────────────────────────────────────────────────────────
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--h-dim", type=int, default=64,
                   help="Horizon-embedding dimension.")
    p.add_argument("--time-emb-dim", type=int, default=32,
                   help="Per-observation time-offset embedding dimension.")
    p.add_argument("--d-model", type=int, default=256,
                   help="Internal dimension for Mamba + LSTM temporal module.")
    p.add_argument("--mamba-layers", type=int, default=3)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.20)

    # ── training ─────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=8.0,
                   help="BCE positive-class weight (VE foreground is spatially sparse).")
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-augment", action="store_true")

    # ── inference ────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--ve-ckpt", type=Path, default=None,
                   help="VE UNet checkpoint — needed when --infer-image is provided.")
    p.add_argument("--infer-image", type=Path, default=None,
                   help="Optional current satellite image; VE extracted via --ve-ckpt.")
    p.add_argument("--infer-date", type=str, default=None,
                   help="Shooting date of the current observation (YYYY-MM).")
    p.add_argument("--infer-aoi", type=str, default=None)
    p.add_argument("--infer-horizon", type=int, default=2,
                   help="Forecast horizon in YEARS (1–5).")
    p.add_argument("--mc-samples", type=int, default=50)
    p.add_argument("--infer-out", type=Path, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# 1.  Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_aoi_year_month(name: str) -> tuple[str, int, int] | None:
    """Extract (aoi, year, month) from a filename stem."""
    m = _AOI_DATE_RE.search(name)
    if not m:
        return None
    aoi = m.group(1).lower()
    year, month = int(m.group(2)), int(m.group(3))
    if not (1 <= month <= 12):
        return None
    return aoi, year, month


def resize_mask(arr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize a float32 binary mask [H, W] to [target_size, target_size].

    Uses LANCZOS resampling (anti-aliased area averaging) followed by
    thresholding at 0.30. This preserves thin-line geometry better than
    NEAREST applied to the final binary mask, and is used consistently
    across the prepare, training, and inference stages.

    Handles non-square inputs without aspect-ratio distortion: the image is
    first padded to a square with zeros, resized, then the pad contribution
    is removed by the threshold step.
    """
    if arr.shape[0] == target_size and arr.shape[1] == target_size:
        return arr

    H, W = arr.shape
    if H != W:
        # Pad shorter axis so LANCZOS doesn't distort the line orientation
        side = max(H, W)
        padded = np.zeros((side, side), dtype=np.float32)
        padded[:H, :W] = arr
        arr = padded

    pil = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (target_size, target_size), Image.LANCZOS
    )
    return (np.asarray(pil, dtype=np.float32) / 255.0 >= 0.30).astype(np.float32)


# ---------------------------------------------------------------------------
# 2.  Mask rendering from LabelMe JSON
# ---------------------------------------------------------------------------
def render_ve_mask_from_json(
    json_path: Path,
    target_size: int,
    line_width: int,
    label: str = "ve",
) -> np.ndarray | None:
    """
    Render VE polylines from a LabelMe JSON to a binary float32 mask [H, W].

    Each annotation shape is rendered independently at the original image
    resolution, then downsampled to target_size via resize_mask().
    Multi-part annotations are OR-merged; points are never merged across shapes.
    """
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    iw = int(payload.get("imageWidth", target_size))
    ih = int(payload.get("imageHeight", target_size))
    shapes = [
        s for s in (payload.get("shapes") or [])
        if str(s.get("label", "")).strip().lower() == label.lower()
        and len(s.get("points") or []) >= 2
    ]
    if not shapes:
        return None

    # Render at native resolution (more accurate line geometry)
    mask_img = Image.new("L", (iw, ih), 0)
    draw = ImageDraw.Draw(mask_img)
    for shape in shapes:
        pts = [(float(p[0]), float(p[1])) for p in shape["points"]]
        draw.line(pts, fill=255, width=max(1, line_width))

    arr = np.asarray(mask_img, dtype=np.float32) / 255.0
    return resize_mask(arr, target_size)


# ---------------------------------------------------------------------------
# 3.  Mask preparation: LabelMe JSON → NPY cache
# ---------------------------------------------------------------------------
def prepare_masks(args: argparse.Namespace) -> None:
    """Discover LabelMe JSONs, render masks, save as NPY files."""
    args.masks_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(args.labelme_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files under {args.labelme_dir}")

    saved = skipped = already = 0
    for jf in json_files:
        result = parse_aoi_year_month(jf.stem)
        if result is None:
            continue
        aoi, year, month = result

        out_dir = args.masks_dir / aoi
        out_dir.mkdir(parents=True, exist_ok=True)
        npy_path = out_dir / f"{year:04d}_{month:02d}.npy"

        if npy_path.exists():
            already += 1
            continue

        mask = render_ve_mask_from_json(jf, target_size=args.mask_size,
                                        line_width=args.line_width)
        if mask is None:
            skipped += 1
            continue

        np.save(npy_path, mask)
        saved += 1
        if (saved + already) % 100 == 0:
            print(f"  saved={saved}  skipped={skipped}  already={already}")

    print(f"Done — saved: {saved}, skipped: {skipped}, already cached: {already}")


# ---------------------------------------------------------------------------
# 4.  Time-series loading  (no gap-filling; real observations only)
# ---------------------------------------------------------------------------
@dataclass
class AOITimeSeries:
    """Sparse, irregularly sampled VE observation sequence for one AOI."""
    aoi: str
    timestamps: list[int]       # year*12 + month, sorted ascending
    masks: list[np.ndarray]     # aligned with timestamps


def load_time_series(masks_dir: Path) -> dict[str, AOITimeSeries]:
    """Load per-AOI mask time series from cached NPY files — real obs only."""
    series: dict[str, AOITimeSeries] = {}

    for aoi_dir in sorted(masks_dir.iterdir()):
        if not aoi_dir.is_dir():
            continue
        aoi = aoi_dir.name.lower()

        records: list[tuple[int, np.ndarray]] = []
        for npy in sorted(aoi_dir.glob("*.npy")):
            m = re.match(r"^(\d{4})_(\d{2})$", npy.stem)
            if not m:
                continue
            year, month = int(m.group(1)), int(m.group(2))
            records.append((year * 12 + month, np.load(npy).astype(np.float32)))

        if len(records) < 3:
            continue

        records.sort(key=lambda r: r[0])
        ts_list = [r[0] for r in records]
        mask_list = [r[1] for r in records]

        series[aoi] = AOITimeSeries(aoi=aoi, timestamps=ts_list, masks=mask_list)
        y0, m0 = ts_list[0] // 12, ts_list[0] % 12
        y1, m1 = ts_list[-1] // 12, ts_list[-1] % 12
        print(f"  {aoi}: {len(ts_list)} obs  [{y0}-{m0:02d} .. {y1}-{m1:02d}]")

    return series


# ---------------------------------------------------------------------------
# 5.  Dataset  (irregular-time-aware, calendar-based target matching)
# ---------------------------------------------------------------------------

# Each pre-built index entry stores lightweight references — no mask copying.
@dataclass(frozen=True)
class _SampleRef:
    ts: AOITimeSeries
    hist_indices: tuple[int, ...]   # indices into ts.timestamps (oldest first)
    horizon_months: int             # forecast horizon
    target_idx: int                 # index of the matched future observation


def _build_padding(
    real_masks: np.ndarray,    # [n_real, H, W]
    real_offsets: np.ndarray,  # [n_real]
    history_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad history to history_len at the front (oldest positions).
    Returns (history [T,H,W], offsets [T], valid_mask [T] bool).
    Padded slots get zero masks and _PAD_OFFSET_MONTHS offset.
    """
    n_real = len(real_offsets)
    n_pad = history_len - n_real

    if n_pad <= 0:
        valid = np.ones(history_len, dtype=bool)
        return real_masks[-history_len:], real_offsets[-history_len:], valid

    pad_mask = np.zeros((n_pad, *real_masks.shape[1:]), dtype=np.float32)
    pad_offs = np.full(n_pad, _PAD_OFFSET_MONTHS, dtype=np.float32)
    history = np.concatenate([pad_mask, real_masks], axis=0)
    offsets = np.concatenate([pad_offs, real_offsets], axis=0)
    valid = np.concatenate([np.zeros(n_pad, dtype=bool),
                             np.ones(n_real, dtype=bool)], axis=0)
    return history, offsets, valid


class VEForecastDataset(Dataset):
    """
    Irregular-time-series dataset for year-conditioned VE forecasting.

    A training sample is constructed as follows:
      1. Choose a reference observation (ref_idx in the AOI's sorted list).
      2. Collect the min(history_len, ref_idx+1) most recent observations
         up to and including ref_idx.  If fewer than min_history real
         observations are available, skip this window.
      3. For each requested forecast horizon h (in months), find the future
         observation whose timestamp is closest to ref_ts + h, within
         ±target_tolerance months.  If no such observation exists, skip.
      4. Optionally filter by target timestamp range (for temporal-split val).
      5. The sample carries:
           history_masks    : [T, 1, H, W]  (oldest first, padded with zeros)
           obs_time_offsets : [T]           (months before ref_ts;
                                             _PAD_OFFSET_MONTHS for padded slots)
           valid_mask       : [T]  bool     (True for real observations)
           horizon_months   : scalar
           target_mask      : [1, H, W]
    """

    def __init__(
        self,
        series_map: dict[str, AOITimeSeries],
        history_len: int,
        min_history: int,
        horizon_months_list: list[int],
        target_tolerance: int,
        augment: bool = False,
        min_target_ts: int | None = None,
        max_target_ts: int | None = None,
    ):
        self.history_len = history_len
        self.augment = augment
        self._refs: list[_SampleRef] = []

        for ts in series_map.values():
            n = len(ts.timestamps)
            for ref_idx in range(min_history - 1, n):
                ref_ts = ts.timestamps[ref_idx]

                # History: up to history_len most recent observations ≤ ref_idx
                hist_start = max(0, ref_idx - history_len + 1)
                hist_indices = tuple(range(hist_start, ref_idx + 1))

                for h_mo in horizon_months_list:
                    target_ts = ref_ts + h_mo

                    # Find closest future observation within tolerance
                    best_idx: int | None = None
                    best_dist = target_tolerance + 1
                    for fi in range(ref_idx + 1, n):
                        dist = abs(ts.timestamps[fi] - target_ts)
                        if dist <= target_tolerance and dist < best_dist:
                            best_dist = dist
                            best_idx = fi

                    if best_idx is None:
                        continue

                    actual_target_ts = ts.timestamps[best_idx]
                    if min_target_ts is not None and actual_target_ts < min_target_ts:
                        continue
                    if max_target_ts is not None and actual_target_ts > max_target_ts:
                        continue

                    self._refs.append(_SampleRef(
                        ts=ts,
                        hist_indices=hist_indices,
                        horizon_months=h_mo,
                        target_idx=best_idx,
                    ))

        print(f"Dataset: {len(self._refs)} samples  "
              f"(history_len={history_len}, min_history={min_history})")

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int):
        ref = self._refs[idx]
        ts = ref.ts
        h_idx = ref.hist_indices
        ref_ts = ts.timestamps[h_idx[-1]]

        real_offsets = np.array(
            [ref_ts - ts.timestamps[i] for i in h_idx], dtype=np.float32
        )
        real_masks = np.stack([ts.masks[i] for i in h_idx], axis=0)  # [n_real, H, W]
        target = ts.masks[ref.target_idx].copy()                       # [H, W]

        history, offsets, valid = _build_padding(real_masks, real_offsets, self.history_len)

        if self.augment:
            history, target = _augment_sequence(history, target)

        return (
            torch.from_numpy(history[:, None]).float(),            # [T, 1, H, W]
            torch.from_numpy(offsets).float(),                     # [T]
            torch.from_numpy(valid),                               # [T] bool
            torch.tensor(float(ref.horizon_months)),               # scalar
            torch.from_numpy(target[None]).float(),                # [1, H, W]
        )


def _augment_sequence(
    history: np.ndarray,   # [T, H, W]
    target: np.ndarray,    # [H, W]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Geometry-consistent spatial augmentation.

    Rules:
    - Flips are applied identically across ALL frames (history + target).
    - Only spatial axes (H, W) are ever reversed — never the temporal axis T.
    - Per-frame morphological noise simulates UNet prediction jitter on
      individual observations, but is NOT applied to the target (we do not
      want to corrupt the supervision signal).
    - No zeroing of timesteps: that changes the semantic meaning of the
      sequence and conflicts with the time-offset conditioning.
    """
    # Horizontal flip (flip W axis: axis 2 for history [T,H,W], axis 1 for target [H,W])
    if random.random() < 0.5:
        history = history[:, :, ::-1].copy()
        target = target[:, ::-1].copy()

    # Vertical flip (flip H axis: axis 1 for history, axis 0 for target)
    if random.random() < 0.3:
        history = history[:, ::-1, :].copy()
        target = target[::-1, :].copy()

    # Per-frame binary dilation / erosion on history only (not target)
    if _HAS_SCIPY and random.random() < 0.35:
        for t in range(history.shape[0]):
            if history[t].sum() < 1:        # skip padded (zero) frames
                continue
            if random.random() < 0.5:
                history[t] = binary_dilation(
                    history[t] > 0.5, iterations=random.randint(1, 2)
                ).astype(np.float32)
            else:
                history[t] = binary_erosion(
                    history[t] > 0.5, iterations=1
                ).astype(np.float32)

    return history, target


# ---------------------------------------------------------------------------
# 6.  Model components
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for arbitrary-shape scalar tensors.
    Works for both [B] (horizon) and [B, T] (observation time offsets).
    Frequency tensor is computed once in __init__ and stored as a buffer.
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalEmbedding requires even dim"
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)   # [half], moves to device with model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: any shape  →  output: (*x.shape, dim)
        orig = x.shape
        x_flat = x.reshape(-1).float()
        args = x_flat[:, None] * self.freqs[None, :]                   # [N, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # [N, dim]
        return emb.reshape(*orig, self.dim)


class HorizonEncoder(nn.Module):
    """Encodes a forecast horizon (months, float) to a [h_dim] vector."""

    def __init__(self, h_dim: int, sin_dim: int = 64):
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, h_dim * 2),
            nn.GELU(),
            nn.Linear(h_dim * 2, h_dim),
        )

    def forward(self, horizon_months: torch.Tensor) -> torch.Tensor:
        # horizon_months: [B]
        return self.mlp(self.sin_emb(horizon_months))               # [B, h_dim]


class SpatialEncoder(nn.Module):
    """
    Encodes a single VE binary mask [1, H, W] → [latent_dim].
    Weights are shared across all timesteps.
    Contains Dropout so MC inference is supported.
    """

    def __init__(self, latent_dim: int, mask_size: int, dropout: float = 0.2):
        super().__init__()
        n_stages = max(1, int(math.log2(mask_size // 4)))
        channels = [1] + [min(32 * (2 ** i), 256) for i in range(n_stages)]

        conv_layers: list[nn.Module] = []
        for i in range(n_stages):
            conv_layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(4)
        flat_dim = channels[-1] * 4 * 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.conv(x)))     # [B, latent_dim]


class MambaLiteBlock(nn.Module):
    """
    Mamba-style temporal mixer: depthwise conv + gated MLP + residual.
    Adapted from segmentation_and_prediction/main.py.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=d_model,
        )
        self.pw1 = nn.Linear(d_model, 8 * d_model)
        self.pw2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = self.ln(x)
        # Causal depthwise conv: trim to original length (removes future padding)
        h = self.dwconv(h.transpose(1, 2))[:, :, :T].transpose(1, 2)
        a, b = self.pw1(h).chunk(2, dim=-1)
        return x + self.drop(self.pw2(torch.tanh(a) * torch.sigmoid(b)))


class TemporalModule(nn.Module):
    """
    Processes a sequence of spatial latents conditioned on:
      - the real observation time offset of each token (irregular spacing)
      - the requested forecast horizon
      - valid_mask [B, T]: padded slots are zeroed after in_proj and each MambaLiteBlock

    Pipeline: project → MambaLite × N → LSTM → context vector.
    """

    def __init__(
        self,
        latent_dim: int,
        h_dim: int,
        time_emb_dim: int,
        d_model: int,
        mamba_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Input token = spatial latent || time embedding || horizon embedding
        token_dim = latent_dim + time_emb_dim + h_dim
        self.time_emb = SinusoidalEmbedding(time_emb_dim)
        self.in_proj = nn.Linear(token_dim, d_model)
        self.mamba = nn.ModuleList([
            MambaLiteBlock(d_model, dropout=dropout) for _ in range(mamba_layers)
        ])
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        frame_latents: torch.Tensor,     # [B, T, latent_dim]
        obs_time_offsets: torch.Tensor,  # [B, T]  months before reference (≥ 0)
        h_emb: torch.Tensor,             # [B, h_dim]
        valid_mask: torch.Tensor | None = None,  # [B, T] bool; None = all valid
    ) -> torch.Tensor:                   # [B, latent_dim]
        B, T, _ = frame_latents.shape

        # Per-token time embedding  [B, T, time_emb_dim]
        time_embs = self.time_emb(obs_time_offsets)

        # Broadcast horizon embedding across the time axis
        h_exp = h_emb.unsqueeze(1).expand(-1, T, -1)               # [B, T, h_dim]

        x = torch.cat([frame_latents, time_embs, h_exp], dim=-1)   # [B, T, token_dim]
        x = self.in_proj(x)                                         # [B, T, d_model]

        # Zero padded tokens so they do not pollute Mamba convolutions or LSTM state
        if valid_mask is not None:
            gate = valid_mask.unsqueeze(-1).to(dtype=x.dtype)      # [B, T, 1]
            x = x * gate

        for blk in self.mamba:
            x = blk(x)
            if valid_mask is not None:
                x = x * gate                                        # re-zero after each block

        out, _ = self.lstm(x)
        return self.out_proj(out[:, -1, :])                         # [B, latent_dim]


class SpatialDecoder(nn.Module):
    """
    Decodes [latent_dim + h_dim] → [1, mask_size, mask_size].
    Dropout2d is present for MC inference.
    """

    def __init__(self, latent_dim: int, h_dim: int, mask_size: int, dropout: float = 0.2):
        super().__init__()
        n_up = max(1, int(math.log2(mask_size // 4)))
        start_ch = min(256, 32 * (2 ** (n_up - 1)))

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim + h_dim, start_ch * 4 * 4),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        ch = start_ch
        for _ in range(n_up):
            out_ch = max(8, ch // 2)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout * 0.5),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 1, 1))     # raw logits
        self.deconv = nn.Sequential(*layers)
        self._start_ch = start_ch

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self._start_ch, 4, 4)
        return self.deconv(h)               # [B, 1, mask_size, mask_size]


class VETemporalForecaster(nn.Module):
    """
    End-to-end VE temporal forecaster.

    Forward inputs:
        history_masks    : [B, T, 1, H, W]
        obs_time_offsets : [B, T]  months before the reference observation (≥ 0;
                           _PAD_OFFSET_MONTHS for padded slots)
        horizon_months   : [B]     requested forecast horizon in months
        valid_mask       : [B, T]  bool — True for real observations (optional)
    Output:
        logits           : [B, 1, H, W]  (apply sigmoid for probabilities)
    """

    def __init__(
        self,
        mask_size: int = 128,
        history_len: int = 12,
        latent_dim: int = 256,
        h_dim: int = 64,
        time_emb_dim: int = 32,
        d_model: int = 256,
        mamba_layers: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.mask_size = mask_size
        self.history_len = history_len
        self.latent_dim = latent_dim

        self.spatial_enc = SpatialEncoder(latent_dim, mask_size, dropout)
        self.horizon_enc = HorizonEncoder(h_dim)
        self.temporal = TemporalModule(
            latent_dim, h_dim, time_emb_dim, d_model,
            mamba_layers, lstm_hidden, lstm_layers, dropout,
        )
        self.spatial_dec = SpatialDecoder(latent_dim, h_dim, mask_size, dropout)

    def forward(
        self,
        history_masks: torch.Tensor,             # [B, T, 1, H, W]
        obs_time_offsets: torch.Tensor,          # [B, T]
        horizon_months: torch.Tensor,            # [B]
        valid_mask: torch.Tensor | None = None,  # [B, T] bool
    ) -> torch.Tensor:
        B, T, C, H, W = history_masks.shape

        frames = history_masks.view(B * T, C, H, W)
        latents = self.spatial_enc(frames).view(B, T, self.latent_dim)

        h_emb = self.horizon_enc(horizon_months)                              # [B, h_dim]
        context = self.temporal(latents, obs_time_offsets, h_emb, valid_mask) # [B, latent_dim]

        logits = self.spatial_dec(torch.cat([context, h_emb], dim=-1))        # [B,1,H,W]

        # Safety: upsample if decoder output differs from input spatial size
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear",
                                   align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# 7.  Loss & metrics
# ---------------------------------------------------------------------------

def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
    return bce_weight * bce + (1.0 - bce_weight) * dice.mean()


def iou_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thr: float = 0.5,
) -> np.ndarray:
    preds = (torch.sigmoid(logits) >= thr).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return ((tp + 1e-6) / (tp + fp + fn + 1e-6)).cpu().numpy()


def _boundary_metrics(
    pred_bin: np.ndarray,   # [H, W] binary
    gt_bin: np.ndarray,     # [H, W] binary
) -> dict[str, float]:
    """
    Average Symmetric Surface Distance (ASSD) and 95th-percentile Hausdorff
    Distance (HD95) between a predicted and ground-truth binary line mask.
    Returns NaN if scipy is unavailable or either mask is empty.
    """
    if not _HAS_SCIPY or pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return {"assd": float("nan"), "hd95": float("nan")}

    dt_gt   = distance_transform_edt(~gt_bin.astype(bool))
    dt_pred = distance_transform_edt(~pred_bin.astype(bool))

    d_pred_to_gt = dt_gt[pred_bin.astype(bool)]
    d_gt_to_pred = dt_pred[gt_bin.astype(bool)]

    assd = (d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0
    hd95 = float(max(np.percentile(d_pred_to_gt, 95),
                     np.percentile(d_gt_to_pred, 95)))
    return {"assd": float(assd), "hd95": hd95}


def _transect_error(
    pred_prob: np.ndarray,   # [H, W] probability map
    gt_bin: np.ndarray,      # [H, W] binary ground truth
) -> dict[str, float]:
    """
    Per-column (cross-shore transect) VE position error in pixels.
    VE position = row index of the maximum value in that column.
    Columns where gt_bin has no foreground are skipped.
    """
    _, W = pred_prob.shape
    errors: list[float] = []
    for col in range(W):
        if gt_bin[:, col].sum() == 0:
            continue
        gt_row  = float(np.argmax(gt_bin[:, col]))
        pred_row = float(np.argmax(pred_prob[:, col]))
        errors.append(abs(gt_row - pred_row))

    if not errors:
        return {"transect_mae": float("nan"), "transect_rmse": float("nan")}

    errs = np.array(errors, dtype=np.float32)
    return {
        "transect_mae":  float(errs.mean()),
        "transect_rmse": float(np.sqrt((errs ** 2).mean())),
    }


# ---------------------------------------------------------------------------
# 8.  Training
# ---------------------------------------------------------------------------

def build_dataloaders(
    series_map: dict[str, AOITimeSeries],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Build train / val / holdout DataLoaders.

    Validation strategy:
    - If --val-aois is specified, those AOIs are reserved exclusively for
      validation (AOI-split mode). Missing val AOIs raise RuntimeError.
    - If --val-aois is omitted (default), a temporal split is used:
      the last val_frac of the global calendar time range forms the validation
      target window; the earlier portion is used for training targets.
      Both sets share the full series for historical context.
    """
    holdout_set = {a.lower() for a in (args.holdout_aois or [])}
    candidates = {k: v for k, v in series_map.items() if k not in holdout_set}
    holdout_series = {k: v for k, v in series_map.items() if k in holdout_set}

    if not candidates:
        raise RuntimeError("No AOIs remain after reserving holdout AOIs.")

    horizon_months = [y * 12 for y in args.horizon_years]
    augment = not args.no_augment
    pin = device.type == "cuda"

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, pin_memory=pin,
            drop_last=(shuffle and len(ds) >= args.batch_size),
        )

    common_kw = dict(
        history_len=args.history_len,
        min_history=args.min_history,
        horizon_months_list=horizon_months,
        target_tolerance=args.target_tolerance,
    )

    val_set = {a.lower() for a in args.val_aois} if args.val_aois else set()

    if val_set:
        # ── AOI-split mode ──────────────────────────────────────────────────
        missing_val = val_set - set(candidates.keys())
        if missing_val:
            raise RuntimeError(
                f"Validation AOI(s) not found in {args.masks_dir}: {sorted(missing_val)}.\n"
                "Run --mode prepare first, or adjust --val-aois."
            )
        train_series = {k: v for k, v in candidates.items() if k not in val_set}
        val_series   = {k: v for k, v in candidates.items() if k in val_set}

        if not train_series:
            raise RuntimeError("No training AOIs remain after reserving val/holdout.")

        train_ds = VEForecastDataset(train_series, augment=augment, **common_kw)
        val_ds   = VEForecastDataset(val_series,   augment=False,   **common_kw)
        print(f"AOI split: train={sorted(train_series)}, val={sorted(val_series)}")

    else:
        # ── Temporal-split fallback ─────────────────────────────────────────
        all_ts_vals = sorted(set(
            ts for s in candidates.values() for ts in s.timestamps
        ))
        if len(all_ts_vals) < 4:
            raise RuntimeError("Too few observations for a temporal split.")

        cutoff_idx = max(1, int(len(all_ts_vals) * (1.0 - args.val_frac)))
        cutoff_ts  = all_ts_vals[cutoff_idx]
        cutoff_yr, cutoff_mo = cutoff_ts // 12, cutoff_ts % 12
        print(f"Temporal split: train targets before {cutoff_yr}-{cutoff_mo:02d}, "
              f"val targets from {cutoff_yr}-{cutoff_mo:02d}  "
              f"(val_frac={args.val_frac})")

        train_ds = VEForecastDataset(
            candidates, augment=augment, max_target_ts=cutoff_ts - 1, **common_kw
        )
        val_ds = VEForecastDataset(
            candidates, augment=False, min_target_ts=cutoff_ts, **common_kw
        )

    if len(val_ds) == 0:
        print("WARNING: validation dataset is empty "
              "(val AOIs/window may not have enough observations for the "
              "requested horizons).")

    holdout_loader = None
    if holdout_series:
        h_ds = VEForecastDataset(holdout_series, augment=False, **common_kw)
        holdout_loader = _loader(h_ds, shuffle=False)

    return _loader(train_ds, True), _loader(val_ds, False), holdout_loader


@torch.no_grad()
def evaluate(
    model: VETemporalForecaster,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    horizon_years: list[int],
    boundary_metrics: bool = False,
) -> dict[str, float]:
    """
    Evaluate model; return aggregate and per-horizon IoU breakdown.
    If boundary_metrics=True, also computes ASSD, HD95, and transect error
    (requires scipy; skipped gracefully if unavailable).
    """
    model.eval()
    all_losses: list[float] = []
    by_horizon: dict[int, list[float]] = {}   # horizon_months → list[IoU]
    bd_metrics: dict[str, list[float]] = {"assd": [], "hd95": [],
                                           "transect_mae": [], "transect_rmse": []}

    for history, offsets, valid, horizon, target in loader:
        history = history.to(device)
        offsets = offsets.to(device)
        valid   = valid.to(device)
        horizon = horizon.to(device)
        target  = target.to(device)

        logits = model(history, offsets, horizon, valid)
        all_losses.append(
            float(dice_bce_loss(logits, target, pos_weight=pos_weight).item())
        )
        sample_ious = iou_per_sample(logits, target)
        for i, h_mo in enumerate(horizon.cpu().numpy()):
            key = int(round(h_mo))
            by_horizon.setdefault(key, []).append(float(sample_ious[i]))

        if boundary_metrics and _HAS_SCIPY:
            probs_np = torch.sigmoid(logits)[:, 0].cpu().numpy()   # [B, H, W]
            gt_np    = target[:, 0].cpu().numpy()                   # [B, H, W]
            for b in range(probs_np.shape[0]):
                pred_bin = (probs_np[b] >= 0.5).astype(bool)
                gt_bin   = gt_np[b].astype(bool)
                bm = _boundary_metrics(pred_bin, gt_bin)
                te = _transect_error(probs_np[b], gt_bin)
                for k, v in {**bm, **te}.items():
                    bd_metrics[k].append(v)

    if not all_losses:
        return {"loss": float("nan"), "iou": float("nan")}

    metrics: dict[str, float] = {
        "loss": float(np.mean(all_losses)),
        "iou":  float(np.mean([v for vs in by_horizon.values() for v in vs])),
    }
    for yr in horizon_years:
        key = yr * 12
        if key in by_horizon:
            metrics[f"iou_{yr}yr"] = float(np.mean(by_horizon[key]))

    if boundary_metrics:
        for k, vs in bd_metrics.items():
            finite = [v for v in vs if np.isfinite(v)]
            metrics[k] = float(np.mean(finite)) if finite else float("nan")

    return metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VE mask time series from {args.masks_dir} …")
    series_map = load_time_series(args.masks_dir)
    if not series_map:
        raise RuntimeError(f"No mask series found in {args.masks_dir}. "
                           "Run --mode prepare first.")

    train_loader, val_loader, holdout_loader = build_dataloaders(
        series_map, args, device
    )

    model = VETemporalForecaster(
        mask_size=args.mask_size,
        history_len=args.history_len,
        latent_dim=args.latent_dim,
        h_dim=args.h_dim,
        time_emb_dim=args.time_emb_dim,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Training on {device} | epochs={args.epochs} | batch={args.batch_size} "
          f"| horizons={args.horizon_years} yr")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6,
    )
    pos_weight = torch.tensor([args.pos_weight], device=device)

    best_iou = -1.0
    no_improve = 0
    best_path = args.output_dir / "ve_forecaster_best.pth"
    last_path = args.output_dir / "ve_forecaster_last.pth"
    history_log_path = args.output_dir / "train_history.jsonl"
    # Start fresh
    history_log_path.write_text("", encoding="utf-8")

    args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    base_ckpt = {
        "args": args_dict,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for history, offsets, valid, horizon, target in train_loader:
            history = history.to(device)
            offsets = offsets.to(device)
            valid   = valid.to(device)
            horizon = horizon.to(device)
            target  = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(history, offsets, horizon, valid)
            loss = dice_bce_loss(logits, target, pos_weight=pos_weight)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_m = evaluate(model, val_loader, device, pos_weight, args.horizon_years)
        scheduler.step(val_m["iou"])

        iou_detail = "  ".join(
            f"{k}={v:.3f}" for k, v in val_m.items() if k.startswith("iou_")
        )
        print(f"Epoch {epoch:03d} | train={train_loss:.4f} | "
              f"val_iou={val_m['iou']:.4f} | {iou_detail} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        rec = {"epoch": epoch, "train_loss": train_loss, **val_m,
               "lr": float(optimizer.param_groups[0]["lr"])}
        # Append a single JSON record per line (no O(epoch²) checkpoint growth)
        with history_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        ckpt = {**base_ckpt,
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_iou": best_iou}
        torch.save(ckpt, last_path)

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            ckpt["best_val_iou"] = best_iou
            torch.save(ckpt, best_path)
            print(f"  → best saved (IoU={best_iou:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stop: no improvement for {no_improve} epochs.")
                break

    # Final holdout evaluation with full metrics
    if holdout_loader is not None and best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        h_m = evaluate(model, holdout_loader, device, pos_weight, args.horizon_years,
                        boundary_metrics=True)
        print("Holdout: " + " | ".join(f"{k}={v:.4f}" for k, v in h_m.items()))

    print(f"Done. Best val IoU: {best_iou:.4f}")


# ---------------------------------------------------------------------------
# 9.  MC Dropout inference
# ---------------------------------------------------------------------------

def enable_mc_dropout(model: nn.Module) -> None:
    """Activate only Dropout/Dropout2d layers for stochastic inference."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@dataclass
class UncertaintyOutput:
    central: np.ndarray       # [H, W] mean probability
    uncertainty: np.ndarray   # [H, W] std over MC samples
    env_low: np.ndarray       # [H, W] 10th percentile
    env_high: np.ndarray      # [H, W] 90th percentile
    binary: np.ndarray        # [H, W] thresholded central estimate
    horizon_years: int
    n_samples: int
    samples: np.ndarray | None = field(default=None, repr=False)


@torch.no_grad()
def mc_predict(
    model: VETemporalForecaster,
    history_masks: np.ndarray,       # [T, H, W]
    obs_time_offsets: np.ndarray,    # [T]  months before reference
    horizon_years: int,
    device: torch.device,
    n_samples: int = 50,
    threshold: float = 0.5,
    return_samples: bool = False,
) -> UncertaintyOutput:
    """N stochastic forward passes → central estimate + uncertainty maps."""
    model.eval()
    enable_mc_dropout(model)

    T, H, W = history_masks.shape
    hist_t  = torch.from_numpy(history_masks[:, None].astype(np.float32)).unsqueeze(0).to(device)
    off_t   = torch.from_numpy(obs_time_offsets.astype(np.float32)).unsqueeze(0).to(device)
    valid_t = (off_t < _PAD_OFFSET_MONTHS)           # [1, T] bool
    h_t     = torch.tensor([float(horizon_years * 12)], device=device)

    # Pre-allocate to avoid per-sample list growth
    probs = np.empty((n_samples, H, W), dtype=np.float32)
    for i in range(n_samples):
        logits = model(hist_t, off_t, h_t, valid_t)
        probs[i] = torch.sigmoid(logits)[0, 0].cpu().numpy()

    central    = probs.mean(axis=0)
    uncertainty = probs.std(axis=0)
    env_low    = np.percentile(probs, 10, axis=0)
    env_high   = np.percentile(probs, 90, axis=0)

    return UncertaintyOutput(
        central=central,
        uncertainty=uncertainty,
        env_low=env_low,
        env_high=env_high,
        binary=(central >= threshold).astype(np.float32),
        horizon_years=horizon_years,
        n_samples=n_samples,
        samples=probs if return_samples else None,
    )


def _ve_line_to_geojson(
    central: np.ndarray,
    uncertainty: np.ndarray,
    env_low: np.ndarray,
    env_high: np.ndarray,
    horizon_years: int,
    n_samples: int,
) -> dict:
    """
    Build a GeoJSON FeatureCollection of VE polylines in pixel coordinates.
    Coordinates are [column, row] (x, y convention).
    Three features are returned: central estimate, 10th-pct, and 90th-pct envelopes.
    """
    W = central.shape[1]
    cols = list(range(W))

    def _argmax_coords(arr: np.ndarray) -> list[list[int]]:
        return [[c, int(np.argmax(arr[:, c]))] for c in cols]

    central_coords = _argmax_coords(central)
    low_coords     = _argmax_coords(env_low)
    high_coords    = _argmax_coords(env_high)

    # Per-column uncertainty at the central VE position
    unc_at_ve = float(np.mean([
        uncertainty[int(np.argmax(central[:, c])), c] for c in cols
    ]))

    return {
        "type": "FeatureCollection",
        "crs_note": "pixel coordinates — [column, row] from top-left origin",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": central_coords},
                "properties": {
                    "label": "ve_central",
                    "horizon_years": horizon_years,
                    "n_mc_samples": n_samples,
                    "uncertainty_mean_at_ve": round(unc_at_ve, 6),
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": low_coords},
                "properties": {
                    "label": "ve_env_low_10pct",
                    "horizon_years": horizon_years,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": high_coords},
                "properties": {
                    "label": "ve_env_high_90pct",
                    "horizon_years": horizon_years,
                },
            },
        ],
    }


def save_uncertainty_outputs(
    result: UncertaintyOutput,
    out_dir: Path,
    metadata: dict | None = None,
) -> None:
    """Save NPY arrays, PNG visualisations, GeoJSON VE line, and metadata JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "central_prob.npy",     result.central)
    np.save(out_dir / "uncertainty.npy",      result.uncertainty)
    np.save(out_dir / "env_low.npy",          result.env_low)
    np.save(out_dir / "env_high.npy",         result.env_high)
    np.save(out_dir / "binary_prediction.npy", result.binary)
    if result.samples is not None:
        np.save(out_dir / "mc_samples.npy", result.samples)

    # ── GeoJSON VE line ──────────────────────────────────────────────────────
    geojson = _ve_line_to_geojson(
        result.central, result.uncertainty, result.env_low, result.env_high,
        result.horizon_years, result.n_samples,
    )
    (out_dir / "ve_forecast.geojson").write_text(
        json.dumps(geojson, indent=2), encoding="utf-8"
    )

    # ── Forecast metadata JSON ────────────────────────────────────────────────
    meta = {
        "horizon_years":   result.horizon_years,
        "n_mc_samples":    result.n_samples,
        "central_mean":    round(float(result.central.mean()), 6),
        "central_max":     round(float(result.central.max()),  6),
        "uncertainty_mean": round(float(result.uncertainty.mean()), 6),
        "uncertainty_max":  round(float(result.uncertainty.max()),  6),
        "envelope_width_mean": round(float((result.env_high - result.env_low).mean()), 6),
        "binary_coverage": round(float(result.binary.mean()), 6),
    }
    if metadata:
        meta.update(metadata)
    (out_dir / "forecast_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    title_base = (f"VE Forecast  |  +{result.horizon_years} yr  "
                  f"|  {result.n_samples} MC samples")

    # ── 4-panel figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title_base, fontsize=13)

    panels = [
        (axes[0, 0], result.central,               "RdYlGn", 0, 1,
         "Central Estimate (mean probability)"),
        (axes[0, 1], result.binary,                "gray",   0, 1,
         "Binary Prediction (threshold = 0.5)"),
        (axes[1, 0], result.uncertainty,           "hot",    0, None,
         "Uncertainty Map (std across MC samples)"),
        (axes[1, 1], result.env_high - result.env_low, "plasma", 0, 1,
         "80 % Confidence Envelope Width (90th − 10th pct)"),
    ]
    for ax, data, cmap, vmin, vmax, ttl in panels:
        vmax = vmax if vmax is not None else (data.max() or 1.0)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "uncertainty_visualization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Column-wise VE position with confidence envelope ────────────────────
    cols = np.arange(result.central.shape[1])
    ve_central = np.argmax(result.central, axis=0)
    ve_low     = np.argmax(result.env_low,  axis=0)
    ve_high    = np.argmax(result.env_high, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cols, ve_central, color="green", lw=2, label="Central VE")
    ax.fill_between(cols, ve_low, ve_high, alpha=0.3, color="green",
                    label="80 % Confidence Envelope")
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (VE position)")
    ax.set_title(f"Spatial Confidence Envelope — Horizon +{result.horizon_years} yr")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "confidence_envelope_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Outputs saved → {out_dir}")


def load_forecaster_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[VETemporalForecaster, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    a = ckpt.get("args", {})

    def _i(k: str, d: int) -> int:     return int(a.get(k, d))
    def _f(k: str, d: float) -> float: return float(a.get(k, d))

    model = VETemporalForecaster(
        mask_size=_i("mask_size", 128),
        history_len=_i("history_len", 12),
        latent_dim=_i("latent_dim", 256),
        h_dim=_i("h_dim", 64),
        time_emb_dim=_i("time_emb_dim", 32),
        d_model=_i("d_model", 256),
        mamba_layers=_i("mamba_layers", 3),
        lstm_hidden=_i("lstm_hidden", 256),
        lstm_layers=_i("lstm_layers", 1),
        dropout=_f("dropout", 0.20),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# ---------------------------------------------------------------------------
# 10.  Inference mode
# ---------------------------------------------------------------------------

def infer(args: argparse.Namespace) -> None:
    """
    Probabilistic VE forecast for a user-selected AOI and horizon.

    The user provides:
      --infer-aoi        AOI identifier
      --infer-date       YYYY-MM  (the reference / most-recent-observation date)
      --infer-horizon    forecast horizon in YEARS (1–5)
      --infer-image      (optional) current satellite image; VE extracted via --ve-ckpt
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint required for inference mode.")
    if args.infer_aoi is None or args.infer_date is None:
        raise ValueError("--infer-aoi and --infer-date are required for inference mode.")

    device = resolve_device(args.device)
    print(f"Loading forecaster: {args.checkpoint}")
    model, ckpt = load_forecaster_checkpoint(args.checkpoint, device)
    saved_args = ckpt.get("args", {})
    mask_size  = int(saved_args.get("mask_size",   128))
    history_len = int(saved_args.get("history_len", 12))

    # Parse reference date
    dm = re.match(r"(\d{4})[_\-](\d{2})", args.infer_date)
    if not dm:
        raise ValueError(f"Cannot parse --infer-date '{args.infer_date}'. Use YYYY-MM.")
    ref_year, ref_month = int(dm.group(1)), int(dm.group(2))
    ref_ts = ref_year * 12 + ref_month

    # Load cached masks for this AOI
    aoi = args.infer_aoi.lower()
    aoi_dir = args.masks_dir / aoi
    if not aoi_dir.exists():
        raise FileNotFoundError(
            f"No cached masks for AOI '{aoi}' in {args.masks_dir}. "
            "Run --mode prepare first."
        )

    avail: list[tuple[int, np.ndarray]] = []
    for npy in sorted(aoi_dir.glob("*.npy")):
        mm = re.match(r"^(\d{4})_(\d{2})$", npy.stem)
        if not mm:
            continue
        ts_val = int(mm.group(1)) * 12 + int(mm.group(2))
        if ts_val > ref_ts:
            continue
        mask_arr = np.load(npy).astype(np.float32)
        # Ensure consistent spatial size using the shared resize pipeline
        if mask_arr.shape[0] != mask_size or mask_arr.shape[1] != mask_size:
            mask_arr = resize_mask(mask_arr, mask_size)
        avail.append((ts_val, mask_arr))

    # Optionally add a freshly-extracted VE mask from the input image
    if args.infer_image is not None:
        print(f"Extracting VE from: {args.infer_image}")
        current_mask = _extract_ve_mask(args.infer_image, args.ve_ckpt, mask_size, device)
        avail.append((ref_ts, current_mask))

    avail.sort(key=lambda x: x[0])

    if not avail:
        raise RuntimeError(f"No observations found for {aoi} up to {ref_year}-{ref_month:02d}.")

    # Take the most recent history_len observations, pad if fewer
    history_window  = avail[-history_len:]
    history_ts_vals = [ts for ts, _ in history_window]
    history_masks   = np.stack([m for _, m in history_window], axis=0)  # [T, H, W]

    actual_ref_ts = history_ts_vals[-1]
    real_offsets  = np.array(
        [actual_ref_ts - ts for ts in history_ts_vals], dtype=np.float32
    )
    history_masks, time_offsets, _ = _build_padding(
        history_masks, real_offsets, history_len
    )

    print(f"Forecasting {aoi} | ref={ref_year}-{ref_month:02d} "
          f"| horizon={args.infer_horizon} yr | MC samples={args.mc_samples}")
    result = mc_predict(
        model, history_masks, time_offsets,
        horizon_years=args.infer_horizon,
        device=device,
        n_samples=args.mc_samples,
    )

    print(f"Central  — mean={result.central.mean():.4f}  max={result.central.max():.4f}")
    print(f"Uncert.  — mean={result.uncertainty.mean():.4f}  "
          f"max={result.uncertainty.max():.4f}")

    out_dir = args.infer_out or (
        args.output_dir / f"infer_{aoi}_{ref_year}{ref_month:02d}_h{args.infer_horizon}yr"
    )
    meta = {
        "aoi": aoi,
        "reference_date": f"{ref_year}-{ref_month:02d}",
        "infer_horizon_years": args.infer_horizon,
        "checkpoint": str(args.checkpoint),
        "n_history_obs": len(history_window),
    }
    save_uncertainty_outputs(result, out_dir, metadata=meta)


def _extract_ve_mask(
    image_path: Path,
    ve_ckpt_path: Optional[Path],
    mask_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run the VE UNet on a satellite image and return a binary mask [H, W].
    Uses the shared resize_mask pipeline (LANCZOS + threshold 0.30).
    """
    if ve_ckpt_path is None:
        raise ValueError("--ve-ckpt is required when --infer-image is provided.")
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for VE mask extraction (connected component filtering).")

    from src.terra_ugla.models.ve_unet import RobustUNet  # noqa: E402

    RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    ckpt = torch.load(ve_ckpt_path, map_location=device, weights_only=True)
    img_sz   = int((ckpt.get("args") or {}).get("image_size",     512))
    base_ch  = int((ckpt.get("args") or {}).get("base_channels",   64))
    ve_model = RobustUNet(n_channels=3, n_classes=1,
                          base_channels=base_ch, apply_sigmoid=False).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    ve_model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
    ve_model.eval()

    image = Image.open(image_path).convert("RGB").resize((img_sz, img_sz), Image.BILINEAR)
    arr = (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    arr = (arr - RGB_MEAN) / RGB_STD
    with torch.no_grad():
        prob = torch.sigmoid(ve_model(
            torch.from_numpy(arr).unsqueeze(0).to(device)
        ))[0, 0].cpu().numpy()

    # Keep only the largest connected component
    binary = (prob >= 0.5).astype(np.uint8)
    labels, count = nd_label(binary)
    if count > 1:
        sizes = [int((labels == i).sum()) for i in range(1, count + 1)]
        binary = (labels == (np.argmax(sizes) + 1)).astype(np.float32)
    else:
        binary = binary.astype(np.float32)

    # Downsample to mask_size using the shared pipeline
    return resize_mask(binary, mask_size)


# ---------------------------------------------------------------------------
# 11.  Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        print("=== Prepare: LabelMe JSONs → binary mask NPY files ===")
        prepare_masks(args)
    elif args.mode == "train":
        print("=== Train VETemporalForecaster ===")
        train(args)
    elif args.mode == "infer":
        print("=== Inference: probabilistic VE forecast ===")
        infer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
