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


def normalize_image(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - RGB_MEAN) / RGB_STD
    return torch.from_numpy(arr).unsqueeze(0)


def largest_component(binary: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(binary.astype(np.uint8))
    if count <= 1:
        return binary
    sizes = ndimage.sum(binary, labels, index=np.arange(1, count + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    return labels == largest_idx


def ordered_points_from_mask(
    prob: np.ndarray,
    threshold: float,
    target_points: int,
    min_points: int,
    smooth_window: int,
    smooth_polyorder: int,
) -> list[list[float]] | None:
    binary = prob >= float(threshold)
    if int(binary.sum()) < 16:
        return None
    binary = largest_component(binary)
    coords = np.argwhere(binary)
    if coords.shape[0] < 8:
        return None

    # Convert (row, col) -> (x, y) and order points along principal axis.
    xy = np.stack([coords[:, 1], coords[:, 0]], axis=1).astype(np.float32)
    center = xy.mean(axis=0, keepdims=True)
    u, _, vh = np.linalg.svd(xy - center, full_matrices=False)
    axis = vh[0] if vh.shape[0] > 0 else np.array([1.0, 0.0], dtype=np.float32)
    t = ((xy - center) @ axis.reshape(2, 1)).reshape(-1)
    order = np.argsort(t)
    xy_sorted = xy[order]
    t_sorted = t[order]

    t_min, t_max = float(t_sorted[0]), float(t_sorted[-1])
    if abs(t_max - t_min) < 1e-6:
        return None

    n_bins = max(int(min_points), int(target_points))
    bins = np.linspace(t_min, t_max, n_bins + 1)
    sampled: list[np.ndarray] = []
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (t_sorted >= left) & (t_sorted <= right)
        else:
            mask = (t_sorted >= left) & (t_sorted < right)
        if mask.any():
            sampled.append(xy_sorted[mask].mean(axis=0))

    if len(sampled) < int(min_points):
        idx = np.linspace(0, len(xy_sorted) - 1, num=max(int(min_points), len(sampled)), dtype=int)
        sampled = [xy_sorted[i] for i in idx]

    pts = np.stack(sampled, axis=0)
    n = len(pts)
    if n >= 7:
        win = int(smooth_window)
        if win % 2 == 0:
            win -= 1
        win = max(5, min(win, n if n % 2 == 1 else n - 1))
        poly = max(1, min(int(smooth_polyorder), win - 1))
        if win >= poly + 2 and win >= 5:
            pts[:, 0] = savgol_filter(pts[:, 0], window_length=win, polyorder=poly, mode="interp")
            pts[:, 1] = savgol_filter(pts[:, 1], window_length=win, polyorder=poly, mode="interp")

    return [[float(p[0]), float(p[1])] for p in pts]


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
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
            inp = normalize_image(image, image_size=image_size).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                logits = model(inp)
                probs = torch.sigmoid(logits)
                probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
                prob = probs[0, 0].detach().cpu().numpy()

            points = ordered_points_from_mask(
                prob=prob,
                threshold=args.threshold,
                target_points=args.target_points,
                min_points=args.min_points,
                smooth_window=args.smooth_window,
                smooth_polyorder=args.smooth_polyorder,
            )
            if points is None or len(points) < 3:
                records.append({**target, "status": "skip", "reason": "no_valid_polyline", "output_json": ""})
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
            records.append({**target, "status": "ok", "reason": "", "points": len(points), "output_json": str(out_json)})
            print(f"[{idx:4d}/{len(targets)}] OK   {out_stem} | points={len(points)}")
        except Exception as exc:
            records.append({**target, "status": "skip", "reason": f"exception:{exc}", "output_json": ""})
            print(f"[{idx:4d}/{len(targets)}] SKIP {out_stem} | exception")

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

