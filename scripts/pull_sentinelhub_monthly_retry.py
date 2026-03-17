"""Retry monthly Sentinel-2 pulls for specific bad months.

This script is for the workflow:
1) You already pulled one image per month.
2) Some months are unusable (clouds/haze/local blur).
3) Re-pull those months with multiple candidates (e.g. 4 images/month).
4) Manually pick the best preview for annotation.

Example:
  python scripts/pull_sentinelhub_monthly_retry.py ^
    --aoi-id aoi_01 ^
    --bbox -2.888374 56.352126 -2.826576 56.36563 ^
    --months 2016-03 2016-06 2016-08 2016-11 2017-04 ^
    --max-images 4 --max-cloud-pct 25
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image

# Allow running as: python scripts/pull_sentinelhub_monthly_retry.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.terra_ugla.config import initialize_sentinel_hub_config
from src.terra_ugla.services.imagery import download_scene_multiband_tiff, search_scenes


MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _safe_scene_id(scene_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", scene_id).strip("_") or "scene"


def _stretch_to_uint8(rgb: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB array shape (3,H,W), got {rgb.shape}")

    out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
    for c in range(3):
        ch = rgb[c, :, :]
        valid = np.isfinite(ch)
        if not np.any(valid):
            continue
        low = float(np.percentile(ch[valid], lower_pct))
        high = float(np.percentile(ch[valid], upper_pct))
        if high <= low:
            high = low + 1e-6
        scaled = np.clip((ch - low) / (high - low), 0.0, 1.0)
        scaled[~valid] = 0.0
        out[:, :, c] = (scaled * 255.0).astype(np.uint8)
    return out


def _save_preview_png(tif_path: Path, out_png: Path, lower_pct: float, upper_pct: float) -> dict[str, Any]:
    with rasterio.open(tif_path) as src:
        rgb = src.read([3, 2, 1]).astype(np.float32)
        if src.nodata is not None:
            rgb[rgb == src.nodata] = np.nan
        preview = _stretch_to_uint8(rgb, lower_pct=lower_pct, upper_pct=upper_pct)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(out_png)
    return {"preview_path": str(out_png), "size": [int(preview.shape[1]), int(preview.shape[0])]}


def _normalize_month_tokens(raw_tokens: list[str]) -> list[str]:
    tokens: list[str] = []
    for token in raw_tokens:
        for piece in re.split(r"[\s,;]+", token.strip()):
            piece = piece.strip()
            if not piece:
                continue
            if not MONTH_RE.match(piece):
                raise ValueError(f"Invalid month token '{piece}'. Expected YYYY-MM")
            tokens.append(piece)
    return sorted(set(tokens))


def _load_months_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Months file not found: {path}")
    tokens: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(re.split(r"[\s,;]+", line))
    return _normalize_month_tokens(tokens)


def _month_date_range(month_token: str, today: date) -> tuple[str, str] | None:
    year, month = map(int, month_token.split("-"))
    start = date(year, month, 1)
    end_day = calendar.monthrange(year, month)[1]
    end = date(year, month, end_day)
    if start > today:
        return None
    if end > today:
        end = today
    return start.isoformat(), end.isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry pull for selected bad months with multiple candidates.")
    parser.add_argument("--aoi-id", required=True, help="AOI id used in run/report naming.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help="WGS84 bounding box.",
    )
    parser.add_argument("--months", nargs="*", default=[], help="Month list in YYYY-MM, e.g. 2016-03 2016-06")
    parser.add_argument("--months-file", default="", help="Optional text file with YYYY-MM entries.")
    parser.add_argument("--max-images", type=int, default=4, help="Candidates per month.")
    parser.add_argument("--max-cloud-pct", type=int, default=25, help="Max cloud percentage.")
    parser.add_argument("--run-tag", default="retry", help="Tag in run id (default: retry).")
    parser.add_argument("--lower-pct", type=float, default=2.0, help="Preview stretch lower percentile.")
    parser.add_argument("--upper-pct", type=float, default=98.0, help="Preview stretch upper percentile.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip month when retry manifest already exists.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    if not (0.0 <= args.lower_pct < args.upper_pct <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")

    months: list[str] = []
    if args.months:
        months.extend(_normalize_month_tokens(args.months))
    if args.months_file:
        months.extend(_load_months_from_file(Path(args.months_file)))
    months = sorted(set(months))
    if not months:
        raise ValueError("No months provided. Use --months or --months-file")

    _, sentinel_hub_available = initialize_sentinel_hub_config()
    bbox = [float(v) for v in args.bbox]
    today = datetime.now(timezone.utc).date()

    report_rows: list[dict[str, Any]] = []
    monthly_summaries: list[dict[str, Any]] = []

    for month_token in months:
        date_range = _month_date_range(month_token, today=today)
        if date_range is None:
            report_rows.append(
                {
                    "aoi_id": args.aoi_id,
                    "month": month_token,
                    "run_id": "",
                    "status": "skipped_future_month",
                    "scene_id": "",
                    "scene_datetime": "",
                    "cloud_pct": "",
                    "is_demo": "",
                    "download_error": "",
                    "tif_path": "",
                    "preview_path": "",
                }
            )
            continue
        start_date, end_date = date_range

        run_id = f"{args.aoi_id}_{args.run_tag}_{month_token.replace('-', '')}"
        run_root = Path("data") / "runs" / run_id
        retry_manifest_path = run_root / "retry_manifest.json"
        if args.skip_existing and retry_manifest_path.exists():
            report_rows.append(
                {
                    "aoi_id": args.aoi_id,
                    "month": month_token,
                    "run_id": run_id,
                    "status": "skipped_existing",
                    "scene_id": "",
                    "scene_datetime": "",
                    "cloud_pct": "",
                    "is_demo": "",
                    "download_error": "",
                    "tif_path": "",
                    "preview_path": "",
                }
            )
            continue

        search_cache_id = f"{args.aoi_id}_{args.run_tag}_{month_token.replace('-', '')}"
        scenes = search_scenes(
            aoi_id=search_cache_id,
            bbox_wgs84=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_pct=int(args.max_cloud_pct),
            max_images=int(args.max_images),
            sentinel_hub_available=sentinel_hub_available,
        )

        scene_records: list[dict[str, Any]] = []
        if not scenes:
            report_rows.append(
                {
                    "aoi_id": args.aoi_id,
                    "month": month_token,
                    "run_id": run_id,
                    "status": "no_scene_found",
                    "scene_id": "",
                    "scene_datetime": "",
                    "cloud_pct": "",
                    "is_demo": "",
                    "download_error": "",
                    "tif_path": "",
                    "preview_path": "",
                }
            )
        else:
            for scene in scenes:
                download_meta = download_scene_multiband_tiff(
                    run_id=run_id,
                    aoi_bbox_wgs84=bbox,
                    scene=scene,
                    sentinel_hub_available=sentinel_hub_available,
                )
                tif_path = Path(download_meta["filepath"])
                scene_safe = _safe_scene_id(download_meta.get("scene_id", tif_path.stem))
                preview_path = run_root / "preview" / f"{scene_safe}.png"
                preview_meta = _save_preview_png(
                    tif_path=tif_path,
                    out_png=preview_path,
                    lower_pct=float(args.lower_pct),
                    upper_pct=float(args.upper_pct),
                )

                row = {
                    "aoi_id": args.aoi_id,
                    "month": month_token,
                    "run_id": run_id,
                    "status": "ok",
                    "scene_id": download_meta.get("scene_id", ""),
                    "scene_datetime": download_meta.get("datetime", ""),
                    "cloud_pct": download_meta.get("cloud_pct", ""),
                    "is_demo": bool(download_meta.get("is_demo", False)),
                    "download_error": download_meta.get("download_error", "") or "",
                    "tif_path": str(tif_path),
                    "preview_path": preview_meta["preview_path"],
                }
                report_rows.append(row)
                scene_records.append(
                    {
                        "scene": scene,
                        "download": download_meta,
                        "preview": preview_meta,
                    }
                )

        month_manifest = {
            "aoi_id": args.aoi_id,
            "run_id": run_id,
            "month": month_token,
            "bbox_wgs84": bbox,
            "query": {
                "start_date": start_date,
                "end_date": end_date,
                "max_cloud_pct": int(args.max_cloud_pct),
                "max_images": int(args.max_images),
            },
            "sentinel_hub_available": bool(sentinel_hub_available),
            "scene_count": len(scene_records),
            "scenes": scene_records,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        run_root.mkdir(parents=True, exist_ok=True)
        with retry_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(month_manifest, f, indent=2)

        monthly_summaries.append(
            {
                "month": month_token,
                "run_id": run_id,
                "scene_count": len(scene_records),
                "manifest": str(retry_manifest_path),
            }
        )

    reports_dir = Path("data") / "runs" / "retry_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{args.aoi_id}_{args.run_tag}_{stamp}"
    report_csv = reports_dir / f"{base}.csv"
    report_json = reports_dir / f"{base}.json"

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "aoi_id",
            "month",
            "run_id",
            "status",
            "scene_id",
            "scene_datetime",
            "cloud_pct",
            "is_demo",
            "download_error",
            "tif_path",
            "preview_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report_rows)

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "aoi_id": args.aoi_id,
                "run_tag": args.run_tag,
                "bbox_wgs84": bbox,
                "months_requested": months,
                "max_images": int(args.max_images),
                "max_cloud_pct": int(args.max_cloud_pct),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": report_rows,
                "monthly_summaries": monthly_summaries,
            },
            f,
            indent=2,
        )

    ok_rows = [r for r in report_rows if r["status"] == "ok"]
    skipped_rows = [r for r in report_rows if str(r["status"]).startswith("skipped")]
    fail_rows = [r for r in report_rows if r["status"] not in {"ok"} and not str(r["status"]).startswith("skipped")]
    print(f"Months requested: {len(months)}")
    print(f"Downloaded candidate scenes: {len(ok_rows)}")
    print(f"Skipped rows: {len(skipped_rows)}")
    print(f"Other non-ok rows: {len(fail_rows)}")
    print(f"Report CSV: {report_csv}")
    print(f"Report JSON: {report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
