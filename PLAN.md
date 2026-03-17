## AOI-First COASTGUARD Transfer Plan for TERRA (Sentinel Hub, Waterline+VE, LSTM MVP)

### Summary
Current TERRA workflow in [src/terra_ugla/app.py](e:/Projects/TERRA_UGLA/src/terra_ugla/app.py) is shoreline-first and uses simplified NDVI proxy logic. COASTGUARD in [COASTGUARD/Toolshed/VegetationLine.py](e:/Projects/TERRA_UGLA/COASTGUARD/Toolshed/VegetationLine.py) and [COASTGUARD/Toolshed/Transects.py](e:/Projects/TERRA_UGLA/COASTGUARD/Toolshed/Transects.py) has the extraction/intersection methods you want, but it is tightly coupled to Earth Engine and a heavy stack.
This plan ports the needed algorithms into TERRA in phases, keeps Sentinel Hub as backend, uses AOI polygon input, extracts waterline + VE automatically, intersects on fixed transects, and adds VE/WL LSTM prediction.

### Public API / Interface Changes
1. Add `POST /aoi`  
Request: `{ "name": str, "polygon_latlng": [[lat,lng],...], "close_polygon": bool }`  
Response: `{ "aoi_id": str, "bbox_wgs84": [minLon,minLat,maxLon,maxLat], "utm_epsg": int }`

2. Add `POST /imagery/search`  
Request: `{ "aoi_id": str, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "max_cloud_pct": int, "max_images": int }`  
Response: `{ "scene_count": int, "scenes": [{"scene_id": str, "datetime": iso, "cloud_pct": float}] }`

3. Add `POST /jobs/extract` (async orchestration)  
Request: `{ "aoi_id": str, "scene_ids": [str], "transect_spacing_m": int, "transect_length_m": int, "offshore_ratio": float, "max_dist_ref_m": int }`  
Response: `{ "job_id": str, "status": "queued" }`

4. Add `GET /jobs/<job_id>`  
Response includes phase progress: `imagery_download`, `waterline_extract`, `transect_build`, `ve_extract`, `intersections`, `export`.

5. Add `POST /jobs/predict`  
Request: `{ "run_id": str, "train_split_date": "YYYY-MM-DD"|null, "sequence_len_days": int, "forecast_days": int }`  
Response: `{ "job_id": str }`

6. Add `GET /results/<run_id>/summary` and `GET /results/<run_id>/download?type=geojson|csv|parquet`

7. Deprecate UI reliance on `/save_shoreline` and make it compatibility-only; workflow entry becomes AOI polygon drawing in [static/js/app.js](e:/Projects/TERRA_UGLA/static/js/app.js) and [templates/index.html](e:/Projects/TERRA_UGLA/templates/index.html).

### Implementation Plan
1. Create new modules and stable boundaries.  
Add `services/aoi.py`, `services/imagery.py`, `services/extraction.py`, `services/intersections.py`, `services/prediction.py`, `services/jobs.py`, and `coastguard_port/` for algorithm ports (no direct runtime dependency on original COASTGUARD folder).

2. Implement AOI-first UI and backend validation.  
Replace shoreline polyline drawing with polygon drawing; enforce min 3 vertices; persist AOI GeoJSON under `data/aoi/<aoi_id>.geojson`; compute UTM EPSG from centroid for all metric operations.

3. Upgrade Sentinel Hub ingestion to analysis-grade data.  
Use Catalog search to enumerate scenes; download per-scene GeoTIFF with raw bands `B02,B03,B04,B08,B11` and consistent grid; store per-scene metadata JSON; remove dependence on colorized NDVI PNG for analytics.

4. Port COASTGUARD extraction kernels into TERRA.  
Port and adapt these functions: spectral indices (`nd_index`, `savi`, `rbnd`, `image_std`), classifiers (`classify_image_NN`, `classify_image_NN_shore`), contour routines (`FindShoreContours_WP`, `FindShoreContours_Water`, `process_contours`, `ProcessShoreline`), and threshold utility (`TZValues` / weighted-peaks logic).  
Load model artifacts from `COASTGUARD/Classification/models` in read-only mode and copy required files into a TERRA model path for deployment repeatability.

5. Define fixed-baseline transect strategy.  
From first accepted extracted waterline, generate transects once in projected CRS (meters), then reuse for all scenes; store `data/transects/<run_id>_baseline.geojson`; compute signed cross-shore distances relative to transect midpoint/reference point.

6. Run per-scene extraction pipeline.  
For each scene: cloud/nodata mask, compute NDVI/MNDWI, extract waterline, extract VE, intersect each with fixed transects, save point intersections and per-transect distances; persist per-scene outputs and aggregate timeseries table.

7. Build prediction dataset and LSTM MVP.  
Construct per-transect timeseries with targets `VE_distance_m`, `WL_distance_m`; daily interpolation (PCHIP); sequence windows (default 10 days); LSTM multi-output next-step regression; recursive forecast for requested horizon; store model + metrics + forecasts.

8. Add exports and reproducible manifests.  
Write run manifest with AOI, scenes used, thresholds, classifier versions, CRS, parameters; export GeoJSON for lines/points and CSV/Parquet for timeseries/prediction.

9. Frontend workflow update.  
New step flow: `Draw AOI -> Find Scenes -> Run Extraction -> View Intersections -> Run Prediction -> Export`; show job progress and per-scene acceptance/rejection reasons.

10. Hardening and docs.  
Update README workflow, dependency setup, model file requirements, and troubleshooting; include migration notes from legacy shoreline-first endpoints.

### Test Cases and Scenarios
1. AOI validation tests.  
Reject self-intersecting polygons, too-small AOI, non-closed rings; verify UTM EPSG selection.

2. Imagery ingestion tests.  
Catalog scene search returns deterministic sorted list; downloaded GeoTIFF has expected bands, CRS, transform, and nodata handling.

3. Algorithm parity tests against COASTGUARD fixtures.  
Run small fixture scenes and compare extracted VE/WL contours and intersection distances within tolerance.

4. Geometry and intersection tests.  
Verify transect generation in projected CRS, one-to-many line intersections, multipoint resolution strategy, and signed distance consistency.

5. End-to-end API test.  
`POST /aoi -> /imagery/search -> /jobs/extract -> /results` on fixture AOI; assert output files and schema.

6. Prediction tests.  
No temporal leakage in train/test split; minimum sample handling; deterministic inference path; forecast schema checks.

7. Regression tests for legacy endpoints.  
Old shoreline endpoints still respond (or return explicit deprecation) without breaking app startup.

### Potential Issues (and Mitigation)
1. Dependency mismatch risk is high.  
COASTGUARD depends on Earth Engine/geemap/osgeo/arosics/pyfes/tensorflow-era stack; mitigation: port only required pure-Python/numpy/skimage/geopandas pieces into isolated `coastguard_port`.

2. Current TERRA imagery path is not analysis-grade.  
Existing NDVI path uses colorized PNG and proxy channels, which is scientifically weak; mitigation: switch to raw multispectral GeoTIFF + explicit NDVI/MNDWI arrays.

3. CRS/metric accuracy risk in current TERRA transects.  
Current transects use degree approximations; mitigation: all transect/intersection math in projected CRS (UTM) only.

4. Long-running jobs in Flask request cycle.  
Multi-scene extraction and training will timeout if synchronous; mitigation: async job runner with status polling.

5. Classifier portability/version drift.  
Pickle models can break across sklearn versions; mitigation: pin compatible sklearn version and add model compatibility check at startup.

6. Scene quality and temporal gaps.  
Clouds and sparse scenes can make LSTM unstable; mitigation: quality filters, min-scene thresholds, and “insufficient data” statuses.

7. Environment currently not healthy for tests.  
Baseline `pytest` currently fails during collection due missing `shapely`; mitigation: lock env bootstrap and CI preflight before feature integration.

8. Ambiguous coastline selection in complex AOIs.  
Multiple contours/inlets can produce wrong primary line; mitigation: deterministic contour selection rule + optional user correction tool in phase 2.

### Assumptions and Defaults Locked
1. Migration style: phased MVP, not full one-shot parity.
2. Imagery backend: Sentinel Hub/CDSE only for this phase.
3. Shoreline definition: waterline (wet-dry boundary).
4. Transects: fixed baseline from first accepted extracted waterline.
5. Prediction scope: VE + waterline only, no waves/tides/topography in MVP.
6. Default extraction params: `max_cloud_pct=30`, `transect_spacing_m=100`, `transect_length_m=500`, `offshore_ratio=0.7`, `max_dist_ref_m=150`.
7. Default prediction params: sequence length `10` days, forecast horizon `30` days, skip training when data is below minimum threshold.
