#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a VE-only RobustUNet from LabelMe linestrip annotations.

Default behavior:
- Uses `data/labelme_work` as labeled source.
- Excludes `aoi_09` from train/val (used as holdout).
- Year split: train < 2020, val >= 2020.
- Training images are augmented; val/holdout images are not.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.terra_ugla.models.ve_unet import RobustUNet  # noqa: E402

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    aoi: str
    year: int | None
    month: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VE extraction model from LabelMe linestrip labels.")
    parser.add_argument("--labelme-root", type=Path, default=PROJECT_ROOT / "data" / "labelme_work")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "models" / "ve_unet")
    parser.add_argument("--exclude-aois", nargs="*", default=["aoi_09"])
    parser.add_argument("--val-start-year", type=int, default=2020)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--label-name", type=str, default="ve")
    parser.add_argument("--line-width", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--pos-weight", type=float, default=10.0,
        help="BCE positive-class weight to counter sparse foreground (thin VE lines). "
             "Rule of thumb: ~(background pixels / foreground pixels) for your tile size.",
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=12,
        help="Stop training if val IoU does not improve for this many consecutive epochs.",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Max L2 norm for gradient clipping (0 = disabled).",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable training augmentation (useful for debugging / ablation).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_aoi_year_month(stem: str) -> tuple[str, int | None, int | None]:
    aoi = ""
    year = None
    month = None
    m_aoi = re.match(r"^(aoi_\d+)", stem, re.IGNORECASE)
    if m_aoi:
        aoi = m_aoi.group(1).lower()
    m_ym = re.search(r"_(20\d{2})(\d{2})__", stem)
    if m_ym:
        year = int(m_ym.group(1))
        month = int(m_ym.group(2))
    return aoi, year, month


def resolve_image_path(label_path: Path, payload: dict) -> Path | None:
    image_path = str(payload.get("imagePath", "") or "").strip()
    candidates: list[Path] = []
    if image_path:
        p = Path(image_path)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((label_path.parent / p).resolve())
    candidates.append((label_path.with_suffix(".png")).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_valid_ve_label(payload: dict, label_name: str) -> bool:
    for shape in payload.get("shapes", []) or []:
        if str(shape.get("label", "")).strip().lower() != label_name.lower():
            continue
        points = shape.get("points", []) or []
        if len(points) >= 2:
            return True
    return False


def discover_samples(labelme_root: Path, label_name: str) -> tuple[list[Sample], list[Path]]:
    samples: list[Sample] = []
    missing_images: list[Path] = []
    for label_path in sorted(labelme_root.rglob("*.json")):
        lower = str(label_path).lower()
        if "refline_see" in lower or label_path.name.endswith("_refline.json"):
            continue
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not is_valid_ve_label(payload, label_name=label_name):
            continue
        image_path = resolve_image_path(label_path, payload)
        if image_path is None:
            missing_images.append(label_path)
            continue
        aoi, year, month = parse_aoi_year_month(label_path.stem)
        samples.append(Sample(image_path=image_path, label_path=label_path, aoi=aoi, year=year, month=month))
    return samples, missing_images


def draw_ve_mask(label_path: Path, width: int, height: int, label_name: str, line_width: int) -> np.ndarray:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for shape in payload.get("shapes", []) or []:
        if str(shape.get("label", "")).strip().lower() != label_name.lower():
            continue
        points = []
        for p in shape.get("points", []) or []:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                points.append((float(p[0]), float(p[1])))
        if len(points) >= 2:
            draw.line(points, fill=1, width=max(1, int(line_width)))
    return np.array(mask_img, dtype=np.float32)


def pad_to_square(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Zero-pad the shorter side so image and mask become square.

    Preserves the original content's aspect ratio, which prevents geometric
    distortion of the VE line when resizing to a square target size.
    """
    w, h = image.size
    if w == h:
        return image, mask
    size = max(w, h)
    pad_x = (size - w) // 2
    pad_y = (size - h) // 2
    sq_image = Image.new("RGB", (size, size), (0, 0, 0))
    sq_image.paste(image, (pad_x, pad_y))
    sq_mask = np.zeros((size, size), dtype=np.float32)
    sq_mask[pad_y : pad_y + h, pad_x : pad_x + w] = mask
    return sq_image, sq_mask


def augment_pair(
    image: Image.Image,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Image.Image, np.ndarray]:
    """Randomly augment image (PIL RGB) and mask (HW float32) in lockstep.

    Geometric transforms are applied to both; photometric transforms
    are applied to the image only.
    """
    # --- Horizontal flip ---
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask[:, ::-1].copy()

    # --- Vertical flip ---
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask[::-1, :].copy()

    # --- Random rotation ±15° ---
    if rng.random() < 0.7:
        angle = float(rng.uniform(-15.0, 15.0))
        image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
        # mode "F" supports float32 pixel values; NEAREST keeps binary mask clean
        mask_pil = Image.fromarray(mask)  # auto-detects mode "F" for float32 2D
        mask = np.asarray(mask_pil.rotate(angle, resample=Image.NEAREST, expand=False), dtype=np.float32)

    # --- Random crop (scale 70–100% of the padded square) ---
    if rng.random() < 0.8:
        w, h = image.size  # square at this point
        scale = float(rng.uniform(0.70, 1.0))
        crop_size = max(64, int(w * scale))
        x0 = int(rng.integers(0, max(1, w - crop_size + 1)))
        y0 = int(rng.integers(0, max(1, h - crop_size + 1)))
        image = image.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        mask = mask[y0 : y0 + crop_size, x0 : x0 + crop_size]

    # --- Color jitter: brightness, contrast, saturation (image only) ---
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.7, 1.3)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.7, 1.3)))
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.8, 1.2)))

    # --- Gaussian blur (image only, simulates haze / sensor noise) ---
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 1.5))))

    return image, mask


class VELineDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        label_name: str,
        line_width: int,
        augment: bool = False,
    ):
        self.samples = samples
        self.image_size = int(image_size)
        self.label_name = label_name
        self.line_width = int(line_width)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        w, h = image.size
        mask = draw_ve_mask(
            label_path=sample.label_path,
            width=w,
            height=h,
            label_name=self.label_name,
            line_width=self.line_width,
        )

        # Pad to square first — prevents aspect-ratio distortion on resize
        image, mask = pad_to_square(image, mask)

        # Stochastic augmentation for training splits only
        if self.augment:
            image, mask = augment_pair(image, mask, rng=np.random.default_rng())

        # Resize to model input size
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = image_arr.transpose(2, 0, 1)
        image_arr = (image_arr - RGB_MEAN) / RGB_STD

        mask_arr = (np.asarray(mask_pil, dtype=np.float32) > 0).astype(np.float32)[None, :, :]
        return torch.from_numpy(image_arr), torch.from_numpy(mask_arr)


def split_samples(
    samples: list[Sample],
    exclude_aois: Iterable[str],
    val_start_year: int | None,
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    exclude_set = {a.lower() for a in exclude_aois}
    holdout = [s for s in samples if s.aoi.lower() in exclude_set]
    train_val = [s for s in samples if s.aoi.lower() not in exclude_set]

    train: list[Sample]
    val: list[Sample]
    if val_start_year is not None:
        train = [s for s in train_val if s.year is None or s.year < val_start_year]
        val = [s for s in train_val if s.year is not None and s.year >= val_start_year]
    else:
        train, val = [], []

    if len(train) == 0 or len(val) == 0:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(train_val))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1.0 - float(val_ratio)))
        split_idx = max(1, min(split_idx, len(indices) - 1))
        train = [train_val[i] for i in indices[:split_idx]]
        val = [train_val[i] for i in indices[split_idx:]]

    return train, val, holdout


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1e-6) / (union + 1e-6))
    return (bce_weight * bce) + ((1.0 - bce_weight) * dice.mean())


def calc_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    union = tp + fp + fn
    iou = ((tp + 1e-6) / (union + 1e-6)).mean().item()
    precision = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
    recall = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    # Per-sample F1, then averaged — avoids the harmonic-mean-of-means bias
    f1 = ((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)).mean().item()
    return {"iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metrics: list[dict] = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        if logits.shape != masks.shape:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = dice_bce_loss(logits, masks)
        losses.append(float(loss.item()))
        metrics.append(calc_metrics(logits, masks))
    if not losses:
        return {k: float("nan") for k in ("loss", "iou", "precision", "recall", "f1")}
    return {
        "loss": float(np.mean(losses)),
        "iou": float(np.mean([m["iou"] for m in metrics])),
        "precision": float(np.mean([m["precision"] for m in metrics])),
        "recall": float(np.mean([m["recall"] for m in metrics])),
        "f1": float(np.mean([m["f1"] for m in metrics])),
    }


def print_split_stats(name: str, samples: list[Sample]) -> None:
    by_aoi: dict[str, int] = {}
    by_year: dict[int, int] = {}
    for s in samples:
        by_aoi[s.aoi] = by_aoi.get(s.aoi, 0) + 1
        if s.year is not None:
            by_year[s.year] = by_year.get(s.year, 0) + 1
    print(f"{name}: {len(samples)}")
    print("  AOI:", ", ".join([f"{k}={v}" for k, v in sorted(by_aoi.items())]) or "-")
    print("  Year:", ", ".join([f"{k}={v}" for k, v in sorted(by_year.items())]) or "-")


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Discovering labeled VE samples...")
    samples, missing_images = discover_samples(args.labelme_root, label_name=args.label_name)
    if not samples:
        print(f"[ERROR] no VE samples found in {args.labelme_root}")
        return 1
    print(f"Total samples: {len(samples)}")
    if missing_images:
        print(f"Missing image pairs: {len(missing_images)} (skipped)")

    train_samples, val_samples, holdout_samples = split_samples(
        samples=samples,
        exclude_aois=args.exclude_aois,
        val_start_year=args.val_start_year,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if len(train_samples) == 0 or len(val_samples) == 0:
        print("[ERROR] empty train/val split.")
        return 1

    print_split_stats("Train", train_samples)
    print_split_stats("Val", val_samples)
    print_split_stats("Holdout (excluded AOIs)", holdout_samples)

    use_augment = not args.no_augment
    print(f"Training augmentation: {'enabled' if use_augment else 'disabled'}")

    train_ds = VELineDataset(
        train_samples, image_size=args.image_size, label_name=args.label_name,
        line_width=args.line_width, augment=use_augment,
    )
    val_ds = VELineDataset(
        val_samples, image_size=args.image_size, label_name=args.label_name,
        line_width=args.line_width, augment=False,
    )
    holdout_ds = (
        VELineDataset(
            holdout_samples, image_size=args.image_size, label_name=args.label_name,
            line_width=args.line_width, augment=False,
        )
        if holdout_samples else None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    holdout_loader = (
        DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if holdout_ds else None
    )

    model = RobustUNet(n_channels=3, n_classes=1, base_channels=args.base_channels, apply_sigmoid=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )

    # Scalar tensor for BCE positive-class weighting — counters sparse VE foreground
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)

    best_iou = -1.0
    no_improve_epochs = 0
    history: list[dict] = []
    best_path = args.output_dir / "ve_robust_unet_best.pth"
    latest_path = args.output_dir / "ve_robust_unet_last.pth"

    print(f"Training on device: {device}  |  pos_weight={args.pos_weight}  |  early_stop={args.early_stop_patience}")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != masks.shape:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = dice_bce_loss(logits, masks, pos_weight=pos_weight)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, val_loader, device=device)
        scheduler.step(val_metrics["iou"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_iou={val_metrics['iou']:.4f} "
            f"| val_f1={val_metrics['f1']:.4f} | lr={record['lr']:.2e}"
        )

        # Stringify Path objects so the checkpoint is safe to load with weights_only=True
        safe_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        ckpt = {
            "model_state_dict": model.state_dict(),
            "args": safe_args,
            "epoch": epoch,
            "best_val_iou": best_iou,
            "image_size": int(args.image_size),
            "line_width": int(args.line_width),
            "label_name": str(args.label_name),
            "history": history,
        }
        torch.save(ckpt, latest_path)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            ckpt["best_val_iou"] = best_iou
            torch.save(ckpt, best_path)
            print(f"  -> saved best checkpoint (IoU={best_iou:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stop_patience:
                print(f"  -> early stopping: no improvement for {no_improve_epochs} epochs.")
                break

    summary = {
        "best_val_iou": best_iou,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "holdout_samples": len(holdout_samples),
        "exclude_aois": [a.lower() for a in args.exclude_aois],
        "val_start_year": args.val_start_year,
        "history": history,
    }

    if holdout_loader is not None and best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
        holdout_metrics = evaluate(model, holdout_loader, device=device)
        summary["holdout_metrics"] = holdout_metrics
        print(
            "Holdout metrics (excluded AOIs): "
            f"IoU={holdout_metrics['iou']:.4f}, F1={holdout_metrics['f1']:.4f}"
        )

    summary_path = args.output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training done. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
