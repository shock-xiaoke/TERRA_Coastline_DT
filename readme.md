# TERRA UGLA - Digital Twin Coastal Prediction

TERRA now runs a **dual-track Digital Twin** workflow:

1. **Synchronous inference loop (front-end request path)**:
   - load focused AOI and latest real coastline status,
   - user enters forecast horizon in years,
   - system predicts and returns forecast coastline GeoJSON immediately.

2. **Asynchronous assimilation loop (background heartbeat)**:
   - periodically poll Sentinel-2 for new low-cloud scenes,
   - run segmentation/extraction pipeline,
   - append new coastline distances to DT state database,
   - periodically retrain prediction model (e.g., every 6 months) when enough samples exist.

## Quick Start

```bash
python -m venv terra_env
terra_env\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open: `http://localhost:5000`

Dedicated baseline workflow:

- `http://localhost:5000/baseline-workflow`
  - Draw coastline baseline manually.
  - Generate and save fixed transects independently from AOI extraction/prediction flow.

## Required Config

Copy `config.example.json` to `config.json` and fill CDSE/Sentinel Hub credentials.

If credentials are missing, scene search/download uses demo fallback rasters so workflow can still run.

## New API Endpoints

### AOI
- `POST /aoi`
  - Request: `{ "name": str, "polygon_latlng": [[lat,lng],...], "close_polygon": bool }`
  - Response: `{ "aoi_id": str, "bbox_wgs84": [minLon,minLat,maxLon,maxLat], "utm_epsg": int }`
- `GET /aoi`
  - List stored AOIs.

### Imagery
- `POST /imagery/search`
  - Request: `{ "aoi_id": str, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "max_cloud_pct": int, "max_images": int }`
  - Response: `{ "scene_count": int, "scenes": [{"scene_id","datetime","cloud_pct"}] }`

### Baseline + Transects
- `POST /baseline`
  - Request: `{ "name": str, "line_latlng": [[lat,lng],...] }`
  - Response: `{ "baseline_id": str, "utm_epsg": int, "bbox_wgs84": [minLon,minLat,maxLon,maxLat] }`
- `GET /baseline`
  - List stored baselines.
- `POST /baseline/<baseline_id>/transects`
  - Request: `{ "spacing_m": float, "transect_length_m": float, "offshore_ratio": float }`
  - Response: `{ "transect_count": int, "geojson": FeatureCollection, ... }`

### Jobs
- `POST /jobs/extract`
  - Request: `{ "aoi_id": str, "scene_ids": [str], "transect_spacing_m": int, "transect_length_m": int, "offshore_ratio": float, "max_dist_ref_m": int }`
  - Response: `{ "job_id": str, "status": "queued" }`
- `POST /jobs/predict`
  - Request: `{ "run_id": str, "train_split_date": "YYYY-MM-DD"|null, "sequence_len_days": int, "forecast_days": int }`
  - Response: `{ "job_id": str, "status": "queued" }`
- `GET /jobs/<job_id>`
  - Returns job state and phase status.
- `POST /jobs/assimilate`
  - Trigger async Digital Twin assimilation cycle (`poll -> extract -> state update -> retrain check`).

### Digital Twin
- `GET /dt/bootstrap`
  - Returns fixed AOI metadata, latest coastline GeoJSON, and DT state summary.
- `POST /dt/predict`
  - Request: `{ "forecast_years": float, "aoi_id": str|null, "sequence_len_days": int, "lookback_days": int }`
  - Response: synchronous forecast GeoJSON reconstructed from model distance outputs.

### Results
- `GET /results/<run_id>/summary`
- `GET /results/<run_id>/download?type=geojson|csv|parquet&artifact=default|forecast|metrics`
  - GeoJSON artifacts: `intersections` (default), `transects`, `waterlines`, `veglines`, `forecast_shorelines`

## Data Outputs

Per run under `data/runs/<run_id>/`:

- `imagery/*.tif`: raw 5-band scene rasters (`B02,B03,B04,B08,B11`)
- `results/timeseries.csv` (+ optional parquet)
- `results/transects.geojson`
- `results/intersections.geojson`
- `exports/waterlines.geojson`
- `exports/veglines.geojson`
- `summary.json`
- `manifest.json`
- `predictions/*` (after prediction job)

Digital Twin state under `data/dt/<aoi_id>/state/`:
- `timeseries.csv` (state database)
- `transects.geojson` (stable reconstruction geometry)
- `latest_waterline.geojson` (latest observed coastline)
- `metadata.json` (last assimilation/retraining info)

## Legacy Endpoints

Old shoreline-first routes are still present for compatibility but are deprecated:

- `/save_shoreline` remains callable (returns `deprecated: true`).
- `/generate_transects`, `/get_satellite_data`, `/analyze_vegetation_edge`, `/extract_vegetation_contours` return explicit deprecation responses.

## Tests

```bash
pytest -q
```

The new tests cover AOI validation, scene search fallback, transect geometry, VE classification basics, prediction artifact generation, and API compatibility.

## Server Training Migration (VE UNet)

This repository version excludes these external components by design:

- `COASTGUARD/`
- `segmentation_and_prediction/`
- `data/` (datasets and model outputs stay outside Git)

### 1) Environment export (local)

Use your local `terra` conda environment to export reproducible snapshots:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/export_env_snapshot.ps1 -CondaEnv terra -OutDir .
```

Generated files:

- `environment.history.yml` (human-managed conda history)
- `requirements.lock.local.txt` (full local pip lock)

Installable base dependency list is tracked in `requirements.in`.

### 2) Dataset packaging and transfer (no Git upload)

`scripts/train_ve_unet.py` requires LabelMe JSON + matching PNG files.

Package locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_labelme_dataset.ps1 -SourceDir data/labelme_work -ArchivePath labelme_work.tar.gz
```

Transfer + extract on server:

```bash
scp labelme_work.tar.gz <user>@<server>:/data/terra/
tar -xzf /data/terra/labelme_work.tar.gz -C /data/terra/
```

### 3) Server setup (Linux + NVIDIA CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

Install PyTorch by CUDA version (example CUDA 12.1):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Install project dependencies:

```bash
pip install -r requirements.in
```

After the server environment is stable, capture a server lockfile:

```bash
python -m pip freeze > requirements.lock.linux-cu121.txt
```

### 4) Train command

```bash
python scripts/train_ve_unet.py \
  --labelme-root /data/terra/labelme_work \
  --output-dir /data/terra/models/ve_unet \
  --device cuda \
  --batch-size 4 \
  --epochs 50 \
  --num-workers 4
```

Expected outputs:

- `ve_robust_unet_best.pth`
- `ve_robust_unet_last.pth`
- `train_summary.json`

### 5) Smoke checks

```bash
python scripts/train_ve_unet.py --help
python scripts/train_ve_unet.py --epochs 1 --batch-size 2 --num-workers 0 --device cuda --labelme-root /data/terra/labelme_work --output-dir /data/terra/models/ve_unet_smoke
```
