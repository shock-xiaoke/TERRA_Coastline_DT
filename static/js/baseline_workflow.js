/* TERRA dedicated baseline + fixed-transect workflow */

let map;
let isDrawingBaseline = false;
let currentBaselinePoints = [];
let currentBaselineId = null;

let baselineMarkerLayer;
let baselineLineLayer;
let transectLayer;

function byId(id) {
  return document.getElementById(id);
}

function updateStatus(targetId, message, type = "info") {
  const el = byId(targetId);
  if (!el) return;
  el.className = `status ${type}`;
  el.textContent = message;
}

function setWorkflowStep(stepId, active) {
  const el = byId(stepId);
  if (!el) return;
  el.classList.toggle("completed", !!active);
}

function initMap() {
  map = L.map("map").setView([56.36, -2.85], 12);

  const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "OpenStreetMap contributors"
  }).addTo(map);

  const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles Esri"
  });

  baselineMarkerLayer = L.layerGroup().addTo(map);
  baselineLineLayer = L.layerGroup().addTo(map);
  transectLayer = L.layerGroup().addTo(map);

  L.control.layers(
    { OpenStreetMap: osm, Satellite: satellite },
    {
      Baseline: baselineLineLayer,
      Transects: transectLayer
    }
  ).addTo(map);
}

function fitMapToLayers() {
  const layers = [];
  [baselineMarkerLayer, baselineLineLayer, transectLayer].forEach((group) => {
    group.eachLayer((layer) => layers.push(layer));
  });
  if (!layers.length) return;
  const bounds = L.featureGroup(layers).getBounds();
  if (bounds.isValid()) {
    map.fitBounds(bounds, { maxZoom: 14 });
  }
}

function clearBaselineDrawing() {
  currentBaselinePoints = [];
  currentBaselineId = null;
  baselineMarkerLayer.clearLayers();
  baselineLineLayer.clearLayers();
  transectLayer.clearLayers();
  byId("saveBaseline").disabled = true;
  byId("generateBaselineTransects").disabled = true;
  updateStatus("baselineInfo", "Baseline cleared.", "info");
  updateStatus("transectInfo", "No transects generated yet.", "info");
}

function addBaselinePoint(e) {
  if (!isDrawingBaseline) return;
  currentBaselinePoints.push([e.latlng.lat, e.latlng.lng]);

  L.circleMarker(e.latlng, {
    radius: 4,
    color: "#222",
    fillColor: "#d81b60",
    fillOpacity: 0.9,
    weight: 1
  }).addTo(baselineMarkerLayer);

  baselineLineLayer.clearLayers();
  if (currentBaselinePoints.length >= 2) {
    L.polyline(currentBaselinePoints, { color: "#d81b60", weight: 3 }).addTo(baselineLineLayer);
    byId("saveBaseline").disabled = false;
  }
}

function startBaselineDrawing() {
  isDrawingBaseline = true;
  byId("startBaseline").disabled = true;
  byId("finishBaseline").disabled = false;
  map.on("click", addBaselinePoint);
  updateStatus("baselineInfo", "Drawing baseline: click coastline points.", "info");
}

function finishBaselineDrawing() {
  isDrawingBaseline = false;
  byId("startBaseline").disabled = false;
  byId("finishBaseline").disabled = true;
  map.off("click", addBaselinePoint);

  baselineLineLayer.clearLayers();
  if (currentBaselinePoints.length >= 2) {
    L.polyline(currentBaselinePoints, { color: "#d81b60", weight: 3 }).addTo(baselineLineLayer);
    byId("saveBaseline").disabled = false;
    setWorkflowStep("bwf1", true);
    updateStatus("baselineInfo", `Baseline complete with ${currentBaselinePoints.length} vertices. Save to continue.`, "success");
  } else {
    updateStatus("baselineInfo", "Baseline needs at least 2 points.", "error");
  }
}

async function saveBaseline() {
  const baselineName = byId("baselineName").value.trim() || `baseline_${new Date().toISOString().slice(0, 19).replace(/[:T-]/g, "")}`;
  if (currentBaselinePoints.length < 2) {
    updateStatus("baselineInfo", "Baseline needs at least 2 points.", "error");
    return;
  }

  try {
    const res = await fetch("/baseline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: baselineName,
        line_latlng: currentBaselinePoints
      })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to save baseline");

    currentBaselineId = data.baseline_id;
    byId("generateBaselineTransects").disabled = false;
    setWorkflowStep("bwf2", true);
    updateStatus("baselineInfo", `Baseline saved: ${currentBaselineId} (UTM ${data.utm_epsg})`, "success");
    await refreshBaselineList();
  } catch (err) {
    updateStatus("baselineInfo", err.message, "error");
  }
}

function drawTransectsGeojson(geojson) {
  transectLayer.clearLayers();
  if (!geojson || !Array.isArray(geojson.features) || geojson.features.length === 0) return;
  L.geoJSON(geojson, {
    style: {
      color: "#6d4c41",
      weight: 1.3,
      opacity: 0.9,
      dashArray: "4,4"
    }
  }).addTo(transectLayer);
}

async function generateBaselineTransects() {
  if (!currentBaselineId) {
    updateStatus("baselineInfo", "Save baseline first.", "error");
    return;
  }

  try {
    const res = await fetch(`/baseline/${encodeURIComponent(currentBaselineId)}/transects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spacing_m: parseFloat(byId("baselineSpacing").value),
        transect_length_m: parseFloat(byId("baselineLength").value),
        offshore_ratio: parseFloat(byId("baselineOffshoreRatio").value)
      })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Could not generate transects");

    drawTransectsGeojson(data.geojson);
    fitMapToLayers();
    setWorkflowStep("bwf3", true);
    updateStatus(
      "transectInfo",
      `Generated ${data.transect_count} transects | offshore ratio=${Number(data.offshore_ratio).toFixed(2)} | onshore/offshore=${Number(data.onshore_length_m).toFixed(1)}/${Number(data.offshore_length_m).toFixed(1)} m`,
      "success"
    );
  } catch (err) {
    updateStatus("transectInfo", err.message, "error");
  }
}

async function refreshBaselineList() {
  try {
    const res = await fetch("/baseline");
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to load baselines");

    const rows = (data.baselines || []).map((b) => {
      return `${b.baseline_id} | UTM ${b.utm_epsg || "?"}`;
    });
    byId("baselineList").textContent = rows.length ? rows.join("\n") : "No baselines loaded.";
  } catch (err) {
    byId("baselineList").textContent = `Failed to load baselines: ${err.message}`;
  }
}

function initEvents() {
  byId("startBaseline").addEventListener("click", startBaselineDrawing);
  byId("finishBaseline").addEventListener("click", finishBaselineDrawing);
  byId("clearBaseline").addEventListener("click", clearBaselineDrawing);
  byId("saveBaseline").addEventListener("click", saveBaseline);
  byId("generateBaselineTransects").addEventListener("click", generateBaselineTransects);
  byId("refreshBaselines").addEventListener("click", refreshBaselineList);
}

function initApp() {
  initMap();
  initEvents();
  refreshBaselineList();
}

document.addEventListener("DOMContentLoaded", initApp);
