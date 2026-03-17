/* DT-only UI: fixed AOI bootstrap + synchronous forecast overlay */

let map;
let currentAoiId = null;
let statusRefreshTimer = null;
let predictionMode = false;

const dtLayers = {
  aoi: null,
  latest: null,
  forecast: null,
  forecastPoints: null
};

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const el = byId(id);
  if (!el) return;
  el.textContent = value;
}

function setStatus(id, message, type = "info") {
  const el = byId(id);
  if (!el) return;
  el.className = `status-line ${type}`;
  el.textContent = message;
}

function setInputError(message = "") {
  const el = byId("forecastInputError");
  if (!el) return;
  el.textContent = message;
}

function formatIso(value) {
  if (!value) return "N/A";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return String(value);
  return dt.toLocaleString();
}

function initMap() {
  map = L.map("map", { zoomControl: true }).setView([56.36, -2.85], 12);

  const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "OpenStreetMap contributors"
  }).addTo(map);

  const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles Esri"
  });

  dtLayers.aoi = L.layerGroup().addTo(map);
  dtLayers.latest = L.layerGroup().addTo(map);
  dtLayers.forecast = L.layerGroup().addTo(map);
  dtLayers.forecastPoints = L.layerGroup().addTo(map);

  L.control.layers(
    { OpenStreetMap: osm, Satellite: satellite },
    {
      "Focused AOI": dtLayers.aoi,
      "Current Coastline": dtLayers.latest,
      "Predicted Coastline": dtLayers.forecast,
      "Predicted Points": dtLayers.forecastPoints
    }
  ).addTo(map);
}

function renderAoiPolygon(points) {
  dtLayers.aoi.clearLayers();
  if (!Array.isArray(points) || points.length < 3) return false;

  try {
    L.polygon(points, {
      color: "#24516e",
      weight: 2,
      fillColor: "#7bb6d9",
      fillOpacity: 0.2
    }).addTo(dtLayers.aoi);
    return true;
  } catch (err) {
    return false;
  }
}

function drawLineGeojson(targetLayer, geojson, styleFn, onEachFeatureFn = null) {
  targetLayer.clearLayers();
  if (!geojson || !Array.isArray(geojson.features) || geojson.features.length === 0) return 0;

  try {
    L.geoJSON(geojson, { style: styleFn, onEachFeature: onEachFeatureFn || undefined }).addTo(targetLayer);
    return geojson.features.length;
  } catch (err) {
    return 0;
  }
}

function lerpHex(startHex, endHex, t) {
  const clamp = Math.max(0, Math.min(1, Number(t)));
  const s = startHex.replace("#", "");
  const e = endHex.replace("#", "");
  const sr = parseInt(s.slice(0, 2), 16);
  const sg = parseInt(s.slice(2, 4), 16);
  const sb = parseInt(s.slice(4, 6), 16);
  const er = parseInt(e.slice(0, 2), 16);
  const eg = parseInt(e.slice(2, 4), 16);
  const eb = parseInt(e.slice(4, 6), 16);
  const r = Math.round(sr + (er - sr) * clamp);
  const g = Math.round(sg + (eg - sg) * clamp);
  const b = Math.round(sb + (eb - sb) * clamp);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

function buildForecastStyleContext(forecastGeojson) {
  const features = getWaterlineForecastFeatures(forecastGeojson);
  const dated = features
    .map((feature) => ({ feature, ts: Date.parse(feature?.properties?.datetime || "") }))
    .filter((entry) => Number.isFinite(entry.ts))
    .sort((a, b) => a.ts - b.ts);

  const byTimestamp = new Map();
  dated.forEach((entry, idx) => {
    byTimestamp.set(entry.ts, idx);
  });

  return {
    datedCount: dated.length,
    byTimestamp
  };
}

function forecastDateRange(forecastGeojson) {
  const features = getWaterlineForecastFeatures(forecastGeojson);
  const times = features
    .map((feature) => Date.parse(feature?.properties?.datetime || ""))
    .filter((ts) => Number.isFinite(ts))
    .sort((a, b) => a - b);
  if (!times.length) return "N/A";
  const first = formatIso(new Date(times[0]).toISOString());
  const last = formatIso(new Date(times[times.length - 1]).toISOString());
  return `${first} to ${last}`;
}

function getWaterlineForecastFeatures(forecastGeojson) {
  const features = Array.isArray(forecastGeojson?.features) ? forecastGeojson.features : [];
  if (!features.length) return [];

  const waterline = features.filter((feature) => {
    const boundaryType = String(feature?.properties?.boundary_type || "").toLowerCase();
    return boundaryType.includes("waterline");
  });
  return waterline.length ? waterline : features;
}

function buildWaterlineForecastGeojson(forecastGeojson) {
  return {
    type: "FeatureCollection",
    features: getWaterlineForecastFeatures(forecastGeojson)
  };
}

function renderForecastPointCloud(forecastGeojson) {
  dtLayers.forecastPoints.clearLayers();
  const features = getWaterlineForecastFeatures(forecastGeojson);
  let pointCount = 0;

  features.forEach((feature) => {
    const props = feature?.properties || {};
    const dateLabel = formatIso(props.datetime);
    const stepLabel = props.forecast_step != null ? `Step ${props.forecast_step}` : "Forecast";
    const coords = Array.isArray(feature?.geometry?.coordinates) ? feature.geometry.coordinates : [];
    coords.forEach((xy) => {
      if (!Array.isArray(xy) || xy.length < 2) return;
      const lon = Number(xy[0]);
      const lat = Number(xy[1]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      L.circleMarker([lat, lon], {
        radius: 2.2,
        color: "#b91c1c",
        weight: 0,
        fillColor: "#f59e0b",
        fillOpacity: 0.85
      })
        .bindPopup(`<strong>${stepLabel}</strong><br>Model point<br>${dateLabel}`)
        .addTo(dtLayers.forecastPoints);
      pointCount += 1;
    });
  });

  return pointCount;
}

function fitMapToLayers() {
  const layers = [];
  Object.values(dtLayers).forEach((group) => {
    group.eachLayer((layer) => layers.push(layer));
  });
  if (!layers.length) return;

  try {
    const bounds = L.featureGroup(layers).getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { maxZoom: 13 });
    }
  } catch (err) {
    // Fail soft on geometry rendering issues.
  }
}

function updateBootstrapPanels(payload) {
  const state = payload.state || {};
  setText("initAoiName", payload.aoi_name || "N/A");
  setText("initAoiId", payload.aoi_id || "N/A");
  setText("initUtm", payload.utm_epsg != null ? String(payload.utm_epsg) : "N/A");
  setText("initLastObs", formatIso(state.last_observation_datetime));

  setText("dtRows", String(Number(state.timeseries_rows || 0)));
  setText("dtLastAssim", formatIso(state.last_assimilation_at));
  setText("dtLastRetrain", formatIso(state.last_retrained_at));
  setText("dtModelRun", state.active_model_run_id || "N/A");
}

async function bootstrapDigitalTwin({ silent = false } = {}) {
  try {
    const response = await fetch("/dt/bootstrap");
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Digital Twin bootstrap failed.");
    }

    currentAoiId = payload.aoi_id;
    byId("runPredict").disabled = !currentAoiId;
    updateBootstrapPanels(payload);

    const polygonOk = renderAoiPolygon(payload.polygon_latlng);
    if (!polygonOk) {
      setStatus("appMessage", "Digital Twin loaded, but AOI polygon is unavailable.", "warn");
    }

    const latestFeatures = predictionMode
      ? 0
      : drawLineGeojson(
        dtLayers.latest,
        payload.latest_waterline_geojson,
        () => ({
          color: "#1f7dbd",
          weight: 2.8,
          opacity: 0.95
        })
      );

    fitMapToLayers();

    if (!silent) {
      if (latestFeatures === 0) {
        setStatus("appMessage", "Digital Twin initialized. No current coastline is available yet.", "warn");
      } else {
        setStatus("appMessage", "Digital Twin initialized and coastline state loaded.", "success");
      }
    }
  } catch (err) {
    byId("runPredict").disabled = true;
    setStatus("appMessage", err.message, "error");
  }
}

function readForecastYears() {
  const raw = String(byId("forecastYears").value || "").trim();
  if (!/^\d+$/.test(raw)) {
    throw new Error("Forecast horizon must be a whole number of years.");
  }
  const years = Number(raw);
  if (!Number.isInteger(years) || years < 1) {
    throw new Error("Forecast horizon must be an integer >= 1.");
  }
  return years;
}

function updatePredictionPanel(result) {
  setText("predRunId", result.run_id || "N/A");
  setText("predModel", result.model_type || "unknown");
  setText("predHorizon", `${result.forecast_years} years (${result.forecast_days} days)`);
  const featureCount = getWaterlineForecastFeatures(result?.forecast_geojson).length;
  setText("predFeatureCount", String(featureCount));
  setText("predDateRange", forecastDateRange(result?.forecast_geojson));
}

async function runPrediction() {
  if (!currentAoiId) {
    setStatus("predMessage", "Prediction unavailable until Digital Twin initialization succeeds.", "error");
    return;
  }

  try {
    setInputError("");
    const forecastYears = readForecastYears();
    byId("runPredict").disabled = true;
    setStatus("predMessage", "Running synchronous inference...", "info");

    const response = await fetch("/dt/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        forecast_years: forecastYears,
        aoi_id: currentAoiId,
        sequence_len_days: 30,
        lookback_days: 730,
        model_preference: "mamba_lstm"
      })
    });
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Prediction request failed.");
    }

    updatePredictionPanel(payload);
    predictionMode = true;

    // Prediction view should visualize only model output coastline.
    dtLayers.latest.clearLayers();

    const rawForecastGeojson = payload.forecast_geojson || { type: "FeatureCollection", features: [] };
    const forecastGeojson = buildWaterlineForecastGeojson(rawForecastGeojson);
    const forecastStyleContext = buildForecastStyleContext(forecastGeojson);
    const featureCount = drawLineGeojson(
      dtLayers.forecast,
      forecastGeojson,
      (feature) => {
        const ts = Date.parse(feature?.properties?.datetime || "");
        const idx = Number.isFinite(ts) ? forecastStyleContext.byTimestamp.get(ts) : null;
        const rank = forecastStyleContext.datedCount > 1 && Number.isFinite(idx)
          ? idx / (forecastStyleContext.datedCount - 1)
          : 0;
        const waterlineColor = lerpHex("#f59e0b", "#b91c1c", rank);
        return {
          color: waterlineColor,
          weight: 2.9,
          opacity: 0.95
        };
      },
      (feature, layer) => {
        const props = feature?.properties || {};
        const dateLabel = formatIso(props.datetime);
        const stepLabel = props.forecast_step != null ? `Step ${props.forecast_step}` : "Forecast";
        layer.bindPopup(`<strong>${stepLabel}</strong><br>Predicted coastline<br>${dateLabel}`);
      }
    );
    const pointCount = renderForecastPointCloud(forecastGeojson);
    fitMapToLayers();

    if (featureCount === 0) {
      setStatus("predMessage", "Prediction completed, but no forecast coastline geometry was returned.", "warn");
    } else {
      setStatus("predMessage", `Prediction completed for ${forecastYears} year(s): ${featureCount} coastline(s), ${pointCount} model points rendered.`, "success");
    }
  } catch (err) {
    if (String(err.message).includes("Run assimilation first")) {
      setStatus("predMessage", "Prediction blocked: Digital Twin has no state yet. Wait for heartbeat assimilation.", "error");
    } else if (String(err.message).includes("whole number")) {
      setInputError(err.message);
      setStatus("predMessage", "Fix input validation errors and retry.", "warn");
    } else {
      setStatus("predMessage", err.message, "error");
    }
  } finally {
    byId("runPredict").disabled = !currentAoiId;
  }
}

function initEvents() {
  byId("runPredict").addEventListener("click", runPrediction);
  byId("forecastYears").addEventListener("input", () => {
    setInputError("");
  });
}

function startStatusRefresh() {
  if (statusRefreshTimer) {
    clearInterval(statusRefreshTimer);
  }
  statusRefreshTimer = setInterval(() => {
    bootstrapDigitalTwin({ silent: true });
  }, 60000);
}

function initApp() {
  initMap();
  initEvents();
  bootstrapDigitalTwin();
  startStatusRefresh();
}

document.addEventListener("DOMContentLoaded", initApp);
