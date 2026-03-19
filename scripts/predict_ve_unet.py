#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch VE extraction with a trained VE RobustUNet checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.terra_ugla.models.ve_unet import RobustUNet  # noqa: E402

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
LABELME_VERSION = "5.11.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VE model inference for monthly AOI preview images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "data" / "runs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "labelme_work" / "ve_pred_2020_2026")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--include-aois", nargs="*", default=[f"aoi_{idx:02d}" for idx in range(1, 9)])
    parser.add_argument("--exclude-aois", nargs="*", default=["aoi_09"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-points", type=int, default=256)
    parser.add_argument("--min-points", type=int, default=64)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--smooth-polyorder", type=int, default=3)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--report-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_to_square(image: Image.Image) -> tuple[Image.Image, int, int, int]:
    """Zero-pad shorter side to make image square.

    Returns (padded_image, pad_x, pad_y, sq_size) so the caller can
    later map model-output coordinates back to original image space.
    """
    w, h = image.size
    sq_size = max(w, h)
    pad_x = (sq_size - w) // 2
    pad_y = (sq_size - h) // 2
    if w == h:
        return image, 0, 0, sq_size
    sq = Image.new("RGB", (sq_size, sq_size), (0, 0, 0))
    sq.paste(image, (pad_x, pad_y))
    return sq, pad_x, pad_y, sq_size


def normalize_image(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - RGB_MEAN) / RGB_STD
    return torch.from_numpy(arr).unsqueeze(0)


def ordered_points_from_mask(
    prob: np.ndarray,
    threshold: float,
    target_points: int,
    min_points: int,
    smooth_window: int,
    smooth_polyorder: int,
) -> tuple[list[list[float]] | None, dict]:
    """Convert a probability map (H, W) to a full-span VE linestrip.

    Orientation-aware strategy:
      1. Detect whether VE runs horizontally or vertically via the
         probability-weighted spatial covariance of the prediction map.
         This requires no threshold and is robust to patchy predictions.
      2. Smooth perpendicular to the detected VE direction (Gaussian σ=1.5)
         to suppress pixel-level noise before argmax.
      3. Per-slice argmax along the dominant axis:
           horizontal → per-column argmax gives one row y per column x
           vertical   → per-row argmax gives one column x per row y
      4. Progressive threshold relaxation: if fewer than min_points slices
         exceed `threshold`, relax to 40 % of threshold, then to the 90th-
         percentile of slice peaks. This ensures a rough line is always
         produced for human correction rather than silently skipping the image.
      5. Interpolate over low-confidence slices so the line spans the full
         image dimension (width for H-VE, height for V-VE).
      6. Savitzky-Golay smooth the resulting 1-D profile.
      7. Downsample evenly to target_points.

    Returns (points, stats) where stats contains orientation, conf_frac,
    and interp_frac for logging and the CSV report.
    """
    H, W = prob.shape
    stats: dict = {"orientation": "?", "conf_frac": 0.0, "interp_frac": 0.0}

    total_mass = float(prob.sum())
    if total_mass < 1e-6:
        return None, stats

    # ── Orientation detection via probability-weighted covariance ─────────────
    # Treats prob as a 2-D mass distribution.  The axis with greater spatial
    # variance is the direction the VE runs along (its "length" axis).
    ys_g, xs_g = np.mgrid[0:H, 0:W]
    mean_y = float((prob * ys_g).sum() / total_mass)
    mean_x = float((prob * xs_g).sum() / total_mass)
    var_y = float((prob * (ys_g - mean_y) ** 2).sum() / total_mass)
    var_x = float((prob * (xs_g - mean_x) ** 2).sum() / total_mass)
    # Horizontal VE: large variance in x, small in y  (line spans columns)
    # Vertical VE:   large variance in y, small in x  (line spans rows)
    horizontal = var_x >= var_y
    stats["orientation"] = "H" if horizontal else "V"

    # ── Per-slice argmax ───────────────────────────────────────────────────────
    # Smooth perpendicular to the VE to stabilise the peak position before argmax.
    #   horizontal: smooth along rows (axis=0) then argmax over rows per column
    #   vertical:   smooth along cols (axis=1) then argmax over cols per row
    smooth_axis = 0 if horizontal else 1
    prob_smooth = ndimage.gaussian_filter1d(prob.astype(np.float64), sigma=1.5, axis=smooth_axis)

    if horizontal:
        slice_argmax = prob_smooth.argmax(axis=0).astype(float)  # (W,): row per col
        slice_max_prob = prob.max(axis=0)                         # (W,)
        n_slices, secondary_max = W, H
    else:
        slice_argmax = prob_smooth.argmax(axis=1).astype(float)  # (H,): col per row
        slice_max_prob = prob.max(axis=1)                         # (H,)
        n_slices, secondary_max = H, W

    # ── Progressive threshold relaxation ──────────────────────────────────────
    thr = float(threshold)
    valid = slice_max_prob >= thr
    if int(valid.sum()) < int(min_points):
        thr = float(threshold) * 0.4
        valid = slice_max_prob >= thr
    if int(valid.sum()) < int(min_points):
        # Last resort: accept slices in the top-10 % of peak probability
        thr = float(np.percentile(slice_max_prob, 90))
        valid = slice_max_prob >= max(thr, 1e-4)
    if int(valid.sum()) < max(4, int(min_points) // 4):
        return None, stats  # image genuinely has no VE signal

    stats["conf_frac"] = round(float(valid.mean()), 3)

    # ── Interpolate over low-confidence slices ─────────────────────────────────
    profile = slice_argmax.copy()
    profile[~valid] = np.nan
    primary_coords = np.arange(n_slices, dtype=float)
    if (~valid).any():
        profile[~valid] = np.interp(
            primary_coords[~valid], primary_coords[valid], profile[valid]
        )
        stats["interp_frac"] = round(float((~valid).mean()), 3)

    # ── Savitzky-Golay smooth ─────────────────────────────────────────────────
    win = int(smooth_window)
    if win % 2 == 0:
        win -= 1
    win = max(5, min(win, n_slices if n_slices % 2 == 1 else n_slices - 1))
    poly = max(1, min(int(smooth_polyorder), win - 1))
    if n_slices >= win and win >= poly + 2 and win >= 5:
        profile = savgol_filter(profile, window_length=win, polyorder=poly, mode="interp")
    profile = np.clip(profile, 0.0, float(secondary_max - 1))

    # ── Downsample evenly to target_points ────────────────────────────────────
    n_out = max(int(min_points), min(int(target_points), n_slices))
    sample_idx = np.linspace(0, n_slices - 1, n_out)
    out_profile = np.interp(sample_idx, primary_coords, profile)
    out_primary = np.linspace(0.0, float(n_slices - 1), n_out)

    if horizontal:
        # primary axis = x (columns 0..W-1), secondary = y (row)
        points = [[float(x), float(y)] for x, y in zip(out_primary, out_profile)]
    else:
        # primary axis = y (rows 0..H-1), secondary = x (column)
        points = [[float(x), float(y)] for x, y in zip(out_profile, out_primary)]

    return points, stats


def write_labelme_json(
    out_json: Path,
    image_name: str,
    image_width: int,
    image_height: int,
    points: list[list[float]],
    threshold: float,
    model_path: Path,
) -> None:
    payload = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [
            {
                "label": "ve",
                "points": points,
                "group_id": None,
                "description": f"predicted by VE RobustUNet; threshold={threshold:.3f}; model={model_path.name}",
                "shape_type": "linestrip",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_manifest_scenes(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    scenes = payload.get("scenes", []) or []
    return [s for s in scenes if isinstance(s, dict)]


def find_preview_for_scene(run_dir: Path, scene: dict[str, Any]) -> tuple[Path | None, str]:
    preview_meta = scene.get("preview", {}) or {}
    rel = str(preview_meta.get("preview_path", "") or "").replace("\\", "/")
    if rel:
        p = (PROJECT_ROOT / rel).resolve()
        if p.exists():
            scene_id = str((scene.get("download", {}) or {}).get("scene_id", p.stem))
            return p, scene_id
    candidates = sorted((run_dir / "preview").glob("*.png"))
    if candidates:
        p = candidates[0]
        scene_id = str((scene.get("download", {}) or {}).get("scene_id", p.stem))
        return p, scene_id
    return None, ""


def iter_targets(
    runs_dir: Path,
    start_year: int,
    end_year: int,
    include_aois: list[str],
    exclude_aois: list[str],
):
    include = {a.lower() for a in include_aois}
    exclude = {a.lower() for a in exclude_aois}
    pattern = re.compile(r"^(aoi_\d+)_(20\d{2})(\d{2})$")

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        m = pattern.match(run_dir.name)
        if not m:
            continue
        aoi = m.group(1).lower()
        year = int(m.group(2))
        month = int(m.group(3))
        if year < int(start_year) or year > int(end_year):
            continue
        if include and aoi not in include:
            continue
        if aoi in exclude:
            continue
        scenes = get_manifest_scenes(run_dir)
        if not scenes:
            continue
        for scene in scenes:
            preview_path, scene_id = find_preview_for_scene(run_dir, scene)
            if preview_path is None:
                continue
            run_id = run_dir.name
            if not scene_id:
                scene_id = preview_path.stem
            yield {
                "run_id": run_id,
                "aoi": aoi,
                "year": year,
                "month": month,
                "scene_id": scene_id,
                "preview_path": preview_path,
            }


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    meta = checkpoint if isinstance(checkpoint, dict) else {}
    image_size = int(meta.get("image_size", 512))
    base_channels = int((meta.get("args", {}) or {}).get("base_channels", 64))
    model = RobustUNet(n_channels=3, n_classes=1, base_channels=base_channels, apply_sigmoid=False).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(state, dict):
        clean_state = {}
        for key, value in state.items():
            clean_state[key.replace("module.", "", 1)] = value
        model.load_state_dict(clean_state, strict=True)
    else:
        raise RuntimeError("Invalid checkpoint format.")
    model.eval()
    return model, {"image_size": image_size, "meta": meta}


def draw_overlay(image_path: Path, points: list[list[float]], out_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        if len(points) >= 2:
            draw.line(points, fill=(255, 64, 64), width=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model, model_meta = load_model(args.checkpoint, device=device)
    image_size = int(model_meta["image_size"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_csv or (args.output_dir / "predict_report.csv")

    targets = list(
        iter_targets(
            runs_dir=args.runs_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            include_aois=args.include_aois,
            exclude_aois=args.exclude_aois,
        )
    )
    if not targets:
        print("[ERROR] no target scenes found.")
        return 1

    print(f"Targets: {len(targets)} | device={device} | checkpoint={args.checkpoint}")
    records: list[dict[str, Any]] = []
    ok_count = 0
    for idx, target in enumerate(targets, start=1):
        run_id = target["run_id"]
        scene_id = target["scene_id"]
        preview_path: Path = target["preview_path"]
        out_stem = f"{run_id}__{scene_id}"
        out_png = args.output_dir / f"{out_stem}.png"
        out_json = args.output_dir / f"{out_stem}.json"
        overlay_path = args.output_dir / "overlays" / f"{out_stem}_overlay.png"

        try:
            image = Image.open(preview_path).convert("RGB")
            w, h = image.size
            # Pad to square — must match the training preprocessing
            sq_image, pad_x, pad_y, sq_size = pad_to_square(image)
            inp = normalize_image(sq_image, image_size=image_size).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                logits = model(inp)
                probs = torch.sigmoid(logits)
                # Resize back to padded-square space, then crop to original dims
                probs = F.interpolate(probs, size=(sq_size, sq_size), mode="bilinear", align_corners=False)
                prob = probs[0, 0, pad_y:pad_y + h, pad_x:pad_x + w].detach().cpu().numpy()

            points, ve_stats = ordered_points_from_mask(
                prob=prob,
                threshold=args.threshold,
                target_points=args.target_points,
                min_points=args.min_points,
                smooth_window=args.smooth_window,
                smooth_polyorder=args.smooth_polyorder,
            )
            if points is None or len(points) < 3:
                records.append({**target, "status": "skip", "reason": "no_valid_polyline",
                                 "output_json": "", **ve_stats})
                print(f"[{idx:4d}/{len(targets)}] SKIP {out_stem} | no_valid_polyline")
                continue

            if args.copy_images:
                shutil.copy2(preview_path, out_png)
                image_name = out_png.name
            else:
                out_png = preview_path
                image_name = str(preview_path)

            write_labelme_json(
                out_json=out_json,
                image_name=image_name,
                image_width=w,
                image_height=h,
                points=points,
                threshold=args.threshold,
                model_path=args.checkpoint,
            )
            draw_overlay(out_png, points, overlay_path)
            ok_count += 1
            records.append({**target, "status": "ok", "reason": "",
                             "points": len(points), "output_json": str(out_json), **ve_stats})
            print(
                f"[{idx:4d}/{len(targets)}] OK   {out_stem} "
                f"| ori={ve_stats['orientation']} conf={ve_stats['conf_frac']:.0%} "
                f"interp={ve_stats['interp_frac']:.0%} pts={len(points)}"
            )
        except Exception as exc:
            import traceback
            records.append({**target, "status": "skip", "reason": f"exception:{exc}", "output_json": ""})
            print(f"[{idx:4d}/{len(targets)}] SKIP {out_stem} | {exc}")
            traceback.print_exc()

    fields = sorted({k for rec in records for k in rec.keys()})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("-" * 72)
    print(f"Predicted: {ok_count}")
    print(f"Skipped  : {len(records) - ok_count}")
    print(f"Report   : {report_path}")
    print(f"Output   : {args.output_dir}")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

