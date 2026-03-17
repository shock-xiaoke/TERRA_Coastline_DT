"""Apply chosen retry candidates to replace original monthly run outputs.

Workflow:
1) You have original monthly runs: data/runs/<aoi_id>_<YYYYMM>/
2) You pulled retry candidates:       data/runs/<aoi_id>_<run_tag>_<YYYYMM>/
3) You manually picked one candidate scene per bad month.
4) This script replaces original month data with the chosen candidate.

CSV format (required columns):
  aoi_id,month,selected_scene_id

Optional columns:
  retry_run_id   (if provided, overrides --run-tag pattern)

Example:
  python scripts/apply_retry_replacements.py \
    --selection-csv data/runs/retry_reports/aoi_01_selected.csv \
    --run-tag retry_bad_months \
    --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _safe_scene_id(scene_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", scene_id).strip("_") or "scene"


def _require_dir(path: Path, hint: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{hint}: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_scene_record(retry_manifest: dict[str, Any], selected_scene_id: str) -> dict[str, Any] | None:
    for item in retry_manifest.get("scenes", []) or []:
        scene_id = item.get("download", {}).get("scene_id", "")
        if scene_id == selected_scene_id:
            return item
    return None


def _collect_files(run_dir: Path, subdir: str, suffixes: tuple[str, ...]) -> list[Path]:
    d = run_dir / subdir
    if not d.exists():
        return []
    out: list[Path] = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(p)
    return sorted(out)


def _backup_existing(original_run_dir: Path, backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    for sub in ["imagery", "preview"]:
        src_dir = original_run_dir / sub
        if src_dir.exists():
            dst_dir = backup_root / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)
    manifest = original_run_dir / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, backup_root / "manifest.json")


def _replace_month(
    aoi_id: str,
    month: str,
    selected_scene_id: str,
    run_tag: str,
    retry_run_id_override: str,
    dry_run: bool,
) -> dict[str, Any]:
    yyyymm = month.replace("-", "")
    original_run_id = f"{aoi_id}_{yyyymm}"
    retry_run_id = retry_run_id_override.strip() if retry_run_id_override.strip() else f"{aoi_id}_{run_tag}_{yyyymm}"

    original_run_dir = Path("data") / "runs" / original_run_id
    retry_run_dir = Path("data") / "runs" / retry_run_id

    _require_dir(original_run_dir, "Original monthly run not found")
    _require_dir(retry_run_dir, "Retry run not found")

    safe_scene = _safe_scene_id(selected_scene_id)
    src_tif = retry_run_dir / "imagery" / f"{safe_scene}.tif"
    src_json = retry_run_dir / "imagery" / f"{safe_scene}.json"
    src_png = retry_run_dir / "preview" / f"{safe_scene}.png"
    for p, name in [(src_tif, "source tif"), (src_json, "source json"), (src_png, "source preview")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    old_imagery = _collect_files(original_run_dir, "imagery", (".tif", ".json"))
    old_preview = _collect_files(original_run_dir, "preview", (".png", ".jpg", ".jpeg"))

    if dry_run:
        return {
            "aoi_id": aoi_id,
            "month": month,
            "status": "dry_run",
            "original_run_id": original_run_id,
            "retry_run_id": retry_run_id,
            "selected_scene_id": selected_scene_id,
            "old_imagery_files": len(old_imagery),
            "old_preview_files": len(old_preview),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = original_run_dir / "_backup_before_retry_replace" / stamp
    _backup_existing(original_run_dir, backup_dir)

    # Remove old monthly outputs (imagery + preview) then copy selected files.
    for p in old_imagery + old_preview:
        p.unlink(missing_ok=True)

    (original_run_dir / "imagery").mkdir(parents=True, exist_ok=True)
    (original_run_dir / "preview").mkdir(parents=True, exist_ok=True)
    dst_tif = original_run_dir / "imagery" / src_tif.name
    dst_json = original_run_dir / "imagery" / src_json.name
    dst_png = original_run_dir / "preview" / src_png.name
    shutil.copy2(src_tif, dst_tif)
    shutil.copy2(src_json, dst_json)
    shutil.copy2(src_png, dst_png)

    # Update original manifest with selected retry scene record.
    retry_manifest_path = retry_run_dir / "retry_manifest.json"
    if retry_manifest_path.exists():
        retry_manifest = _load_json(retry_manifest_path)
        selected_record = _find_scene_record(retry_manifest, selected_scene_id)
    else:
        retry_manifest = {}
        selected_record = None

    original_manifest_path = original_run_dir / "manifest.json"
    if original_manifest_path.exists():
        original_manifest = _load_json(original_manifest_path)
    else:
        original_manifest = {"run_id": original_run_id}

    if selected_record is not None:
        original_manifest["scene_count"] = 1
        original_manifest["scenes"] = [selected_record]
    else:
        # Fallback minimal record
        original_manifest["scene_count"] = 1
        original_manifest["scenes"] = [
            {
                "scene": {"scene_id": selected_scene_id},
                "download": {"scene_id": selected_scene_id, "filepath": str(dst_tif)},
                "preview": {"preview_path": str(dst_png)},
            }
        ]

    original_manifest["selection_applied"] = {
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_retry_run_id": retry_run_id,
        "selected_scene_id": selected_scene_id,
        "backup_dir": str(backup_dir),
    }
    _write_json(original_manifest_path, original_manifest)

    audit_path = original_run_dir / "selection_applied.json"
    _write_json(
        audit_path,
        {
            "aoi_id": aoi_id,
            "month": month,
            "original_run_id": original_run_id,
            "retry_run_id": retry_run_id,
            "selected_scene_id": selected_scene_id,
            "source_files": {"tif": str(src_tif), "json": str(src_json), "preview": str(src_png)},
            "target_files": {"tif": str(dst_tif), "json": str(dst_json), "preview": str(dst_png)},
            "backup_dir": str(backup_dir),
            "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "aoi_id": aoi_id,
        "month": month,
        "status": "replaced",
        "original_run_id": original_run_id,
        "retry_run_id": retry_run_id,
        "selected_scene_id": selected_scene_id,
        "old_imagery_files": len(old_imagery),
        "old_preview_files": len(old_preview),
        "backup_dir": str(backup_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace original monthly images with chosen retry candidates.")
    parser.add_argument("--selection-csv", required=True, help="CSV with columns: aoi_id,month,selected_scene_id")
    parser.add_argument("--run-tag", default="retry_bad_months", help="Retry run tag used in run ids.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print actions without modifying files.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    csv_path = Path(args.selection_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"aoi_id", "month", "selected_scene_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})

    results: list[dict[str, Any]] = []
    failures = 0
    for i, row in enumerate(rows, start=1):
        aoi_id = row.get("aoi_id", "")
        month = row.get("month", "")
        selected_scene_id = row.get("selected_scene_id", "")
        retry_run_id = row.get("retry_run_id", "")
        if not aoi_id or not selected_scene_id or not MONTH_RE.match(month):
            results.append(
                {
                    "row": i,
                    "aoi_id": aoi_id,
                    "month": month,
                    "selected_scene_id": selected_scene_id,
                    "status": "invalid_row",
                    "error": "Need aoi_id + selected_scene_id + month(YYYY-MM)",
                }
            )
            failures += 1
            continue

        try:
            r = _replace_month(
                aoi_id=aoi_id,
                month=month,
                selected_scene_id=selected_scene_id,
                run_tag=args.run_tag,
                retry_run_id_override=retry_run_id,
                dry_run=args.dry_run,
            )
            r["row"] = i
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            results.append(
                {
                    "row": i,
                    "aoi_id": aoi_id,
                    "month": month,
                    "selected_scene_id": selected_scene_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    out_dir = Path("data") / "runs" / "retry_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = out_dir / f"apply_retry_replacements_{stamp}.json"
    out_csv = out_dir / f"apply_retry_replacements_{stamp}.csv"

    _write_json(
        out_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": bool(args.dry_run),
            "selection_csv": str(csv_path),
            "run_tag": args.run_tag,
            "results": results,
        },
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in results for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    replaced = sum(1 for r in results if r.get("status") in {"replaced", "dry_run"})
    print(f"Rows processed: {len(results)}")
    print(f"Rows replaced/dry-run-ok: {replaced}")
    print(f"Rows failed/invalid: {failures}")
    print(f"Result CSV: {out_csv}")
    print(f"Result JSON: {out_json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
