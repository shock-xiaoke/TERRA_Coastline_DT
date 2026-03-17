"""Simple Labelme visualization script (no scoring).

This version is designed for your current dataset that only contains `ve`
annotations, but it can visualize any Labelme shapes if needed.

Example:
  python scripts/visualize_labelme_quality.py \
      --input-dir data/labelme_work/auto_labeled \
      --output-dir data/labelme_work/auto_labeled_vis \
      --label ve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def _find_image(json_path: Path, payload: dict[str, Any]) -> Path | None:
    image_path = payload.get("imagePath")
    if isinstance(image_path, str) and image_path.strip():
        candidate = json_path.parent / image_path
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _color_for_label(label: str) -> tuple[int, int, int]:
    label = label.lower()
    if label == "ve":
        return (50, 220, 80)
    if label == "coastline":
        return (255, 215, 0)
    if label == "water":
        return (40, 140, 255)
    return (255, 120, 40)


def _draw_shapes(
    image_path: Path,
    shapes: list[dict[str, Any]],
    out_path: Path,
    label_filter: str,
    line_width: int,
) -> int:
    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    drawn = 0
    for shp in shapes:
        label = str(shp.get("label", "")).strip()
        if label_filter and label.lower() != label_filter.lower():
            continue

        shape_type = str(shp.get("shape_type", "linestrip")).lower().strip()
        points_raw = shp.get("points", []) or []
        points = [tuple(map(float, p[:2])) for p in points_raw if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(points) < 2:
            continue

        r, g, b = _color_for_label(label)
        if shape_type == "polygon" and len(points) >= 3:
            draw.polygon(points, fill=(r, g, b, 55), outline=(r, g, b, 220))
        elif shape_type in {"line", "linestrip", "polyline"}:
            draw.line(points, fill=(r, g, b, 235), width=line_width)
            for x, y in points[:: max(1, len(points) // 40)]:
                draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(255, 255, 255, 200))
        else:
            # Fallback: connect points as a line.
            draw.line(points, fill=(r, g, b, 235), width=line_width)
        drawn += 1

    merged = Image.alpha_composite(base, overlay)
    text = f"{image_path.name} | drawn_shapes={drawn} | filter={label_filter or 'ALL'}"
    text_draw = ImageDraw.Draw(merged)
    font = ImageFont.load_default()
    text_draw.rectangle((6, 6, min(1000, merged.width - 6), 30), fill=(0, 0, 0, 160))
    text_draw.text((10, 12), text, fill=(255, 255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.convert("RGB").save(out_path)
    return drawn


def _write_html(report_path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Labelme Visualization</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f7f8fa;margin:20px;}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;}",
        "th{background:#222;color:#fff;position:sticky;top:0;}",
        "img{max-width:560px;height:auto;border:1px solid #aaa;background:#fff;}",
        "</style></head><body>",
        "<h2>Labelme Overlay Report</h2>",
        "<table>",
        "<tr><th>Image</th><th>Drawn Shapes</th><th>Overlay</th></tr>",
    ]
    for row in rows:
        overlay = row["overlay_relpath"].replace("\\", "/")
        lines.append(
            f"<tr><td>{row['image_name']}</td><td>{row['drawn_shapes']}</td>"
            f"<td><a href='{overlay}' target='_blank'><img src='{overlay}'></a></td></tr>"
        )
    lines.extend(["</table>", "</body></html>"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize Labelme annotations (overlay only, no scoring).")
    parser.add_argument("--input-dir", default="data/labelme_work/auto_labeled", help="Folder with .json + images.")
    parser.add_argument("--output-dir", default="data/labelme_work/auto_labeled_vis", help="Output folder.")
    parser.add_argument("--label", default="ve", help="Only draw this label. Use empty string for all labels.")
    parser.add_argument("--line-width", type=int, default=2, help="Line width for linestrip/line overlays.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for quick checks (0 = all).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if args.max_images > 0:
        json_files = json_files[: args.max_images]
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 1

    report_rows: list[dict[str, str]] = []
    missing_images = 0

    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        image_path = _find_image(json_path, payload)
        if image_path is None:
            missing_images += 1
            continue

        shapes = payload.get("shapes", []) or []
        out_path = overlay_dir / f"{image_path.stem}_overlay.png"
        drawn = _draw_shapes(
            image_path=image_path,
            shapes=shapes,
            out_path=out_path,
            label_filter=args.label.strip(),
            line_width=max(1, int(args.line_width)),
        )

        report_rows.append(
            {
                "image_name": image_path.name,
                "drawn_shapes": str(drawn),
                "overlay_relpath": str(out_path.relative_to(output_dir)),
            }
        )

    report_path = output_dir / "report.html"
    _write_html(report_path, report_rows)

    print(f"Processed JSON files: {len(json_files)}")
    print(f"Overlay images: {len(report_rows)}")
    print(f"Missing source images: {missing_images}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
