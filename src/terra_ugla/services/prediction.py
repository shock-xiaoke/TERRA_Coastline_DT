"""Prediction service for VE/WL timeseries forecasts with on-the-fly fine-tuning."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from .intersections import transform_geometry

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMBA_MODEL_PATH = _PACKAGE_ROOT / "models" / "ml_models" / "mamba_lstm_model.pth"
LEGACY_MAMBA_MODEL_PATH = Path("segmentation_and_prediction") / "mamba_lstm_model.pth"
BASE_MAMBA_MODEL_PATH = Path("data") / "models" / "mamba_lstm_base.pt"


@dataclass
class PredictionArtifacts:
    summary: dict[str, Any]
    forecast_df: pd.DataFrame
    metrics_df: pd.DataFrame


def resolve_mamba_checkpoint_path(explicit_path: str | None = None) -> Path | None:
    """Resolve runtime checkpoint preference for Mamba-LSTM model."""
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    env_path = os.getenv("TERRA_MAMBA_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    if DEFAULT_MAMBA_MODEL_PATH.exists():
        return DEFAULT_MAMBA_MODEL_PATH
    if LEGACY_MAMBA_MODEL_PATH.exists():
        return LEGACY_MAMBA_MODEL_PATH
    if BASE_MAMBA_MODEL_PATH.exists():
        return BASE_MAMBA_MODEL_PATH
    return None


def resolve_mamba_coastline_checkpoint_path(explicit_path: str | None = None) -> Path | None:
    """
    Resolve checkpoint for coastline-shape MambaLSTM inference.

    This intentionally excludes `BASE_MAMBA_MODEL_PATH` because that file is
    produced by the VE/WL distance regressor and is architecture-incompatible
    with the coastline-shape model trained in `segmentation_and_prediction/main.py`.
    """
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    env_path = os.getenv("TERRA_MAMBA_COASTLINE_MODEL_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    if DEFAULT_MAMBA_MODEL_PATH.exists():
        return DEFAULT_MAMBA_MODEL_PATH
    if LEGACY_MAMBA_MODEL_PATH.exists():
        return LEGACY_MAMBA_MODEL_PATH
    return None


def _extract_state_dict(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    return (
        payload.get("state_dict")
        or payload.get("model_state_dict")
        or payload.get("model")
        or payload
    )


def _build_daily_series(group: pd.DataFrame) -> pd.DataFrame:
    series = group.sort_values("datetime").dropna(subset=["VE_distance_m", "WL_distance_m"]).copy()
    if series.empty:
        return pd.DataFrame(columns=["VE_distance_m", "WL_distance_m"])

    idx = pd.to_datetime(series["datetime"], utc=True)
    data = series[["VE_distance_m", "WL_distance_m"]].astype(float).to_numpy()

    day_index = pd.date_range(idx.min().normalize(), idx.max().normalize(), freq="D", tz="UTC")
    x = idx.view("int64") / 1e9
    x_day = day_index.view("int64") / 1e9

    out = {}
    for col_idx, col_name in enumerate(["VE_distance_m", "WL_distance_m"]):
        y = data[:, col_idx]
        if len(np.unique(x)) < 2:
            out[col_name] = np.full(len(day_index), float(y[-1]))
            continue

        interpolator = PchipInterpolator(x, y, extrapolate=False)
        vals = interpolator(x_day)
        if np.isnan(vals).any():
            vals = pd.Series(vals).interpolate(method="linear", limit_direction="both").to_numpy()
        out[col_name] = vals

    return pd.DataFrame(out, index=day_index)


def _create_sequences(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(seq_len, len(values)):
        xs.append(values[i - seq_len : i, :])
        ys.append(values[i, :])
    if not xs:
        return np.empty((0, seq_len, values.shape[1])), np.empty((0, values.shape[1]))
    return np.asarray(xs), np.asarray(ys)


def _try_import_torch() -> tuple[Any | None, Any | None]:
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except Exception:
        return None, None


def _try_build_mamba(d_model: int):
    try:
        from mamba_ssm import Mamba

        return Mamba(d_model=d_model)
    except Exception:
        return None


def _build_mamba_lstm_regressor(nn, torch, input_dim: int = 2, d_model: int = 64, mixer_layers: int = 2, lstm_hidden: int = 64, dropout: float = 0.1):
    class MambaLiteBlock(nn.Module):
        def __init__(self, block_d_model: int, block_dropout: float = 0.1, kernel_size: int = 3):
            super().__init__()
            self.norm = nn.LayerNorm(block_d_model)
            self.dwconv = nn.Conv1d(
                block_d_model,
                block_d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=block_d_model,
            )
            self.pw1 = nn.Linear(block_d_model, block_d_model * 4)
            self.pw2 = nn.Linear(block_d_model * 2, block_d_model)
            self.drop = nn.Dropout(block_dropout)

        def forward(self, x):
            h = self.norm(x)
            h = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
            u = self.pw1(h)
            a, b = u.chunk(2, dim=-1)
            h = torch.tanh(a) * torch.sigmoid(b)
            h = self.pw2(h)
            return x + self.drop(h)

    class MambaLSTMRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(input_dim, d_model)
            self.pos_bias = nn.Parameter(torch.zeros(1, 512, d_model))
            built = _try_build_mamba(d_model)
            if built is not None:
                self.mixer = nn.ModuleList([_try_build_mamba(d_model) for _ in range(mixer_layers)])
            else:
                self.mixer = nn.ModuleList([MambaLiteBlock(d_model, block_dropout=dropout) for _ in range(mixer_layers)])
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.head = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(lstm_hidden, input_dim))

        def forward(self, x):
            _, t, _ = x.shape
            z = self.in_proj(x) + self.pos_bias[:, :t, :]
            for block in self.mixer:
                z = block(z)
            out, _ = self.lstm(z)
            return self.head(out[:, -1, :])

    return MambaLSTMRegressor()


def _train_tf_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dense(16, activation="relu"),
            Dense(2),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

    callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=min(64, max(8, len(x_train) // 5)),
        verbose=0,
        callbacks=[callback],
    )

    return model, {
        "epochs": len(history.history.get("loss", [])),
        "final_loss": float(history.history.get("loss", [np.nan])[-1]),
        "final_val_loss": float(history.history.get("val_loss", [np.nan])[-1]),
    }


def _train_mamba_lstm_torch(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    pred_dir: Path,
    warm_start_model_path: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    torch, nn = _try_import_torch()
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is not installed.")

    if len(x_train) < 8 or len(x_val) < 2:
        raise RuntimeError("Not enough samples for stable MambaLSTM fine-tuning.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_mamba_lstm_regressor(nn=nn, torch=torch).to(device)

    loaded_checkpoint = False
    warm_path = resolve_mamba_checkpoint_path(warm_start_model_path)
    if warm_path is not None:
        try:
            state = torch.load(warm_path, map_location=device)
            state_dict = _extract_state_dict(state)
            model.load_state_dict(state_dict, strict=False)
            loaded_checkpoint = True
        except Exception:
            loaded_checkpoint = False

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_ds = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=min(64, max(8, len(x_train) // 4)), shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    max_epochs = 60
    patience = 12
    stale = 0
    final_train = np.nan
    final_val = np.nan

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = criterion(val_pred, y_val_t)
            final_val = float(val_loss.detach().cpu().item())
            final_train = float(np.mean(train_losses)) if train_losses else np.nan

        scheduler.step(final_val)

        if final_val < best_val:
            best_val = final_val
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("MambaLSTM fine-tuning failed to converge.")

    model.load_state_dict(best_state)
    pred_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = pred_dir / "mamba_lstm_finetuned.pt"
    torch.save(model.state_dict(), checkpoint_path)

    BASE_MAMBA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), BASE_MAMBA_MODEL_PATH)

    return model, {
        "on_the_fly_finetune": True,
        "loaded_checkpoint": loaded_checkpoint,
        "warm_start_path": str(warm_path) if warm_path is not None else None,
        "epochs": best_epoch,
        "best_val_loss": float(best_val),
        "final_train_loss": float(final_train),
        "final_val_loss": float(final_val),
        "model_checkpoint": str(checkpoint_path),
        "base_checkpoint_updated": str(BASE_MAMBA_MODEL_PATH),
        "device": str(device),
    }


def _predict_mamba_torch(model: Any, x_np: np.ndarray) -> np.ndarray:
    torch, _ = _try_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    device = next(model.parameters()).device
    with torch.no_grad():
        xb = torch.tensor(x_np, dtype=torch.float32, device=device)
        yb = model(xb).detach().cpu().numpy()
    return yb


def _load_mamba_lstm_model(checkpoint_path: str | None) -> tuple[Any | None, dict[str, Any]]:
    resolved = resolve_mamba_checkpoint_path(checkpoint_path)
    if resolved is None:
        return None, {"loaded_checkpoint": False, "reason": "checkpoint_not_found"}

    torch, nn = _try_import_torch()
    if torch is None or nn is None:
        return None, {"loaded_checkpoint": False, "reason": "pytorch_not_available"}

    path = resolved

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_mamba_lstm_regressor(nn=nn, torch=torch).to(device)
        state = torch.load(path, map_location=device)
        state_dict = _extract_state_dict(state)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, {"loaded_checkpoint": True, "path": str(path), "device": str(device)}
    except Exception as exc:
        return None, {"loaded_checkpoint": False, "reason": f"load_error:{exc}"}


def _build_mamba_lstm_coastline_model(
    nn,
    torch,
    *,
    n_points: int = 256,
    history_len: int = 5,
    d_model: int = 256,
    mamba_layers: int = 2,
    lstm_hidden: int = 256,
    lstm_layers: int = 1,
    dropout: float = 0.1,
):
    # Mirrors `MambaLSTMForecaster` in `segmentation_and_prediction/main.py`.
    class MambaLiteBlock(nn.Module):
        def __init__(self, block_d_model: int, block_dropout: float = 0.1, kernel_size: int = 3):
            super().__init__()
            self.ln = nn.LayerNorm(block_d_model)
            self.dwconv = nn.Conv1d(
                block_d_model,
                block_d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=block_d_model,
            )
            self.pw1 = nn.Linear(block_d_model, 8 * block_d_model)
            self.pw2 = nn.Linear(4 * block_d_model, block_d_model)
            self.drop = nn.Dropout(block_dropout)

        def forward(self, x):
            h = self.ln(x)
            h = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
            u = self.pw1(h)
            a, b = u.chunk(2, dim=-1)
            h = torch.tanh(a) * torch.sigmoid(b)
            h = self.pw2(h)
            return x + self.drop(h)

    class MambaLSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            h_dim = n_points * 2
            self.in_proj = nn.Linear(h_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, history_len, d_model) * 0.02)

            real_mamba = _try_build_mamba(d_model)
            if real_mamba is not None:
                self.mamba_stack = nn.ModuleList([_try_build_mamba(d_model) for _ in range(mamba_layers)])
            else:
                self.mamba_stack = nn.ModuleList(
                    [MambaLiteBlock(d_model, block_dropout=dropout) for _ in range(mamba_layers)]
                )

            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.out = nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, h_dim),
            )

        def forward(self, x):
            bsz, t_steps, n_pts, n_coord = x.shape
            xf = x.reshape(bsz, t_steps, n_pts * n_coord)
            z = self.in_proj(xf) + self.pos_enc[:, :t_steps, :]
            for block in self.mamba_stack:
                z = block(z)
            out, _ = self.lstm(z)
            last = out[:, -1, :]
            y = self.out(last)
            return y.reshape(bsz, n_pts, n_coord)

    return MambaLSTMForecaster()


def _load_mamba_coastline_model(
    checkpoint_path: str | None,
    *,
    history_len: int = 5,
    n_points: int = 256,
) -> tuple[Any | None, dict[str, Any]]:
    resolved = resolve_mamba_coastline_checkpoint_path(checkpoint_path)
    if resolved is None:
        return None, {"loaded_checkpoint": False, "reason": "checkpoint_not_found"}

    torch, nn = _try_import_torch()
    if torch is None or nn is None:
        return None, {"loaded_checkpoint": False, "reason": "pytorch_not_available"}

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_mamba_lstm_coastline_model(
            nn,
            torch,
            n_points=int(n_points),
            history_len=int(history_len),
            d_model=256,
            mamba_layers=2,
            lstm_hidden=256,
            lstm_layers=1,
            dropout=0.1,
        ).to(device)
        loaded = torch.load(resolved, map_location=device)
        state_dict = _extract_state_dict(loaded)
        if not isinstance(state_dict, dict):
            return None, {"loaded_checkpoint": False, "reason": "invalid_state_dict"}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, {"loaded_checkpoint": True, "path": str(resolved), "device": str(device)}
    except Exception as exc:
        return None, {"loaded_checkpoint": False, "reason": f"load_error:{exc}"}


def _extract_linestring_from_feature(feature: dict[str, Any]):
    try:
        from shapely.geometry import shape
    except Exception:
        return None

    geometry = feature.get("geometry")
    if geometry is None:
        return None

    try:
        geom = shape(geometry)
    except Exception:
        return None

    if geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length, default=None)
    return None


def _collect_observed_waterlines_for_aoi(
    aoi_id: str,
    latest_waterline_geojson: dict[str, Any],
    lookback_days: int = 730,
) -> list[tuple[pd.Timestamp, Any]]:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(lookback_days)))
    rows: list[tuple[pd.Timestamp, Any]] = []

    runs_root = Path("data") / "runs"
    if runs_root.exists():
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue

            summary_path = run_dir / "summary.json"
            waterlines_path = run_dir / "exports" / "waterlines.geojson"
            if not (summary_path.exists() and waterlines_path.exists()):
                continue

            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                continue

            summary_aoi_id = str(summary.get("aoi_id", "")).strip()
            if summary_aoi_id and summary_aoi_id != aoi_id:
                continue
            if not summary_aoi_id and aoi_id not in run_dir.name:
                continue

            try:
                with waterlines_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            for feature in payload.get("features", []):
                line = _extract_linestring_from_feature(feature)
                if line is None or line.length <= 0:
                    continue
                dt = pd.to_datetime((feature.get("properties") or {}).get("datetime"), utc=True, errors="coerce")
                if pd.isna(dt) or dt < cutoff:
                    continue
                rows.append((dt, line))

    # Ensure the most recent state coastline is always available.
    for feature in latest_waterline_geojson.get("features", []):
        line = _extract_linestring_from_feature(feature)
        if line is None or line.length <= 0:
            continue
        dt = pd.to_datetime((feature.get("properties") or {}).get("datetime"), utc=True, errors="coerce")
        if pd.isna(dt):
            dt = pd.Timestamp.now(tz="UTC")
        rows.append((dt, line))

    if not rows:
        return []

    # Deduplicate by timestamp, keeping the longest line for each timestamp.
    dedup: dict[str, tuple[pd.Timestamp, Any]] = {}
    for dt, line in rows:
        key = dt.isoformat()
        if key not in dedup or line.length > dedup[key][1].length:
            dedup[key] = (dt, line)

    return sorted(dedup.values(), key=lambda item: item[0])


def _resample_line_coords(coords: np.ndarray, n_points: int) -> np.ndarray | None:
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        return None

    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    coords = coords[np.sort(unique_idx)]
    if len(coords) < 2:
        return None

    segment = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    total = float(cumulative[-1])
    if total <= 0:
        return None

    target = np.linspace(0.0, total, int(n_points))
    x_new = np.interp(target, cumulative, coords[:, 0])
    y_new = np.interp(target, cumulative, coords[:, 1])
    return np.column_stack([x_new, y_new]).astype(np.float32)


def _line_to_model_coords(line, bbox_wgs84: tuple[float, float, float, float], n_points: int) -> np.ndarray | None:
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_wgs84]
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)

    coords = np.asarray(line.coords, dtype=np.float32)
    if len(coords) < 2:
        return None

    # Match training preprocessing: coastline points sorted left->right.
    coords = coords[np.argsort(coords[:, 0])]
    resampled = _resample_line_coords(coords, n_points=n_points)
    if resampled is None:
        return None

    out = np.empty_like(resampled, dtype=np.float32)
    out[:, 0] = (resampled[:, 0] - min_lon) / lon_span
    out[:, 1] = (resampled[:, 1] - min_lat) / lat_span
    return np.clip(out, 0.0, 1.0)


def _model_coords_to_line(model_coords: np.ndarray, bbox_wgs84: tuple[float, float, float, float]):
    try:
        from shapely.geometry import LineString
    except Exception:
        return None

    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox_wgs84]
    lon_span = max(max_lon - min_lon, 1e-9)
    lat_span = max(max_lat - min_lat, 1e-9)

    coords = np.clip(model_coords.astype(np.float32), 0.0, 1.0)
    lon = min_lon + coords[:, 0] * lon_span
    lat = min_lat + coords[:, 1] * lat_span
    line = LineString([(float(x), float(y)) for x, y in zip(lon, lat)])
    if line.is_empty or line.length <= 0:
        return None
    return line


def _predict_next_coastline(model: Any, history_seq: np.ndarray) -> np.ndarray:
    torch, _ = _try_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")

    device = next(model.parameters()).device
    with torch.no_grad():
        xb = torch.tensor(history_seq, dtype=torch.float32, device=device).unsqueeze(0)
        yb = model(xb)[0].detach().cpu().numpy()
    return yb.astype(np.float32)


def run_mamba_coastline_prediction(
    *,
    run_id: str,
    aoi_id: str,
    aoi_bbox_wgs84: tuple[float, float, float, float] | list[float],
    latest_waterline_geojson: dict[str, Any],
    forecast_days: int,
    checkpoint_path: str | None = None,
    history_len: int = 5,
    n_points: int = 256,
    lookback_days: int = 730,
    return_all_steps: bool = False,
) -> PredictionArtifacts:
    """
    Coastline-shape inference path aligned with `segmentation_and_prediction/main.py`.

    Input: sequence of historical coastline polylines (resampled to 256 points).
    Output: future coastline polylines in GeoJSON.
    """
    try:
        from shapely.geometry import mapping
    except Exception as exc:
        raise RuntimeError(f"shapely_not_available:{exc}") from exc

    bbox = tuple(float(v) for v in aoi_bbox_wgs84)
    if len(bbox) != 4:
        raise ValueError("Invalid AOI bbox for coastline prediction.")

    run_dir = Path("data") / "runs" / run_id
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    observed = _collect_observed_waterlines_for_aoi(
        aoi_id=aoi_id,
        latest_waterline_geojson=latest_waterline_geojson,
        lookback_days=lookback_days,
    )
    if not observed:
        raise RuntimeError("no_observed_waterline_history")

    history_items: list[tuple[pd.Timestamp, np.ndarray]] = []
    for dt, line in observed:
        arr = _line_to_model_coords(line, bbox_wgs84=bbox, n_points=n_points)
        if arr is None:
            continue
        history_items.append((dt, arr))

    if not history_items:
        raise RuntimeError("failed_to_prepare_history_from_observed_waterlines")

    model, load_info = _load_mamba_coastline_model(
        checkpoint_path=checkpoint_path,
        history_len=int(history_len),
        n_points=int(n_points),
    )
    if model is None:
        reason = load_info.get("reason", "unknown")
        raise RuntimeError(f"mamba_coastline_load_failed:{reason}")

    steps = max(1, int(np.ceil(max(1, int(forecast_days)) / 30.0)))
    last_obs_dt = history_items[-1][0]
    if pd.isna(last_obs_dt):
        last_obs_dt = pd.Timestamp.now(tz="UTC")

    seq = [arr for _, arr in history_items]
    if len(seq) < history_len:
        pad = [seq[0].copy() for _ in range(history_len - len(seq))]
        seq = pad + seq

    step_features: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        window = np.stack(seq[-history_len:], axis=0)
        pred = np.clip(_predict_next_coastline(model, window), 0.0, 1.0)
        seq.append(pred)

        pred_line = _model_coords_to_line(pred, bbox_wgs84=bbox)
        if pred_line is None:
            continue

        pred_dt = (last_obs_dt + pd.DateOffset(months=step)).tz_convert("UTC")
        step_features.append(
            {
                "type": "Feature",
                "properties": {
                    "run_id": run_id,
                    "aoi_id": aoi_id,
                    "datetime": pred_dt.isoformat(),
                    "boundary_type": "waterline_forecast",
                    "model_type": "mamba_lstm_coastline",
                    "forecast_step": int(step),
                    "forecast_step_unit": "month",
                    "history_len": int(history_len),
                },
                "geometry": mapping(pred_line),
            }
        )
        step_rows.append(
            {
                "run_id": run_id,
                "aoi_id": aoi_id,
                "datetime": pred_dt.isoformat(),
                "forecast_step": int(step),
                "forecast_step_unit": "month",
                "boundary_type": "waterline_forecast",
                "model_type": "mamba_lstm_coastline",
                "n_points": int(n_points),
            }
        )

    if return_all_steps:
        features = step_features
        forecast_rows = step_rows
    else:
        # Main model in `segmentation_and_prediction/main.py` predicts one coastline
        # per call. For a forecast horizon, we roll the sequence forward internally
        # and expose only the final single coastline.
        features = step_features[-1:] if step_features else []
        forecast_rows = step_rows[-1:] if step_rows else []

    forecast_geojson = {"type": "FeatureCollection", "features": features}
    with (pred_dir / "shoreline_forecast.geojson").open("w", encoding="utf-8") as f:
        json.dump(forecast_geojson, f, indent=2)

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_csv(pred_dir / "forecast.csv", index=False)
    try:
        forecast_df.to_parquet(pred_dir / "forecast.parquet", index=False)
    except Exception:
        pass

    metrics_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model_type": "mamba_lstm_coastline",
                "history_observations": int(len(history_items)),
                "history_len": int(history_len),
                "forecast_steps": int(steps),
                "forecast_features": int(len(features)),
                "checkpoint_path": str(load_info.get("path", "")),
            }
        ]
    )
    metrics_df.to_csv(pred_dir / "metrics.csv", index=False)

    summary = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "mamba_lstm_coastline",
        "model_preference": "mamba_lstm",
        "allow_training": False,
        "sequence_len_days": int(history_len) * 30,
        "history_len": int(history_len),
        "forecast_days": int(forecast_days),
        "forecast_steps": int(steps),
        "forecast_step_unit": "month",
        "forecast_years": round(float(forecast_days) / 365.25, 3),
        "history_observations": int(len(history_items)),
        "coastline_points": int(n_points),
        "training": {
            "training_skipped": True,
            "loaded_checkpoint": bool(load_info.get("loaded_checkpoint")),
            "checkpoint_path": load_info.get("path"),
            "device": load_info.get("device"),
        },
        "files": {
            "forecast_csv": str(pred_dir / "forecast.csv"),
            "metrics_csv": str(pred_dir / "metrics.csv"),
            "forecast_geojson": str(pred_dir / "shoreline_forecast.geojson"),
        },
    }

    with (pred_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return PredictionArtifacts(summary=summary, forecast_df=forecast_df, metrics_df=metrics_df)


def _build_forecast_shoreline_geojson(
    run_id: str,
    forecast_df: pd.DataFrame,
    transects_path: Path,
    utm_epsg: int,
) -> dict[str, Any]:
    if forecast_df.empty or not transects_path.exists():
        return {"type": "FeatureCollection", "features": []}

    try:
        from shapely.geometry import LineString, mapping, shape
    except Exception:
        return {"type": "FeatureCollection", "features": []}

    with transects_path.open("r", encoding="utf-8") as f:
        transects_geojson = json.load(f)

    transects_utm: dict[int, Any] = {}
    for feature in transects_geojson.get("features", []):
        props = feature.get("properties", {})
        transect_id = props.get("transect_id")
        if transect_id is None:
            continue

        geom_wgs84 = shape(feature.get("geometry"))
        if geom_wgs84.is_empty:
            continue

        geom_utm = transform_geometry(geom_wgs84, 4326, utm_epsg)
        if geom_utm.geom_type == "LineString":
            line_utm = geom_utm
        elif geom_utm.geom_type == "MultiLineString":
            line_utm = max(geom_utm.geoms, key=lambda geom: geom.length, default=None)
        else:
            line_utm = None

        if line_utm is not None and line_utm.length > 0:
            transects_utm[int(transect_id)] = line_utm

    if not transects_utm:
        return {"type": "FeatureCollection", "features": []}

    features = []
    df = forecast_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    for boundary_col, boundary_type in [
        ("future_WL_distance_m", "waterline_forecast"),
        ("future_VE_distance_m", "vegetation_edge_forecast"),
    ]:
        for dt, group in df.groupby("datetime"):
            points: list[tuple[int, Any]] = []
            for row in group.itertuples(index=False):
                transect_id = int(row.transect_id)
                distance = getattr(row, boundary_col, np.nan)
                if pd.isna(distance):
                    continue

                line_utm = transects_utm.get(transect_id)
                if line_utm is None or line_utm.length <= 0:
                    continue

                pos = (line_utm.length / 2.0) + float(distance)
                pos = min(max(pos, 0.0), line_utm.length)
                point_utm = line_utm.interpolate(pos)
                point_wgs84 = transform_geometry(point_utm, utm_epsg, 4326)
                points.append((transect_id, point_wgs84))

            if len(points) < 2:
                continue

            points.sort(key=lambda item: item[0])
            line_wgs84 = LineString([(point.x, point.y) for _, point in points])
            if line_wgs84.length <= 0:
                continue

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "run_id": run_id,
                        "datetime": dt.isoformat(),
                        "boundary_type": boundary_type,
                        "model_type": str(group["model_type"].iloc[0]),
                    },
                    "geometry": mapping(line_wgs84),
                }
            )

    return {"type": "FeatureCollection", "features": features}


def _build_sparse_fallback_forecast(
    run_id: str,
    daily_map: dict[int, pd.DataFrame],
    horizon: int,
    model_type: str,
) -> pd.DataFrame:
    """Forecast from sparse per-transect series using linear drift from the latest step."""
    rows: list[dict[str, Any]] = []
    for transect_id, daily in daily_map.items():
        values = daily[["VE_distance_m", "WL_distance_m"]].to_numpy(dtype=float)
        if len(values) == 0:
            continue

        current_date = daily.index[-1]
        last = values[-1, :]
        slope = (values[-1, :] - values[-2, :]) if len(values) >= 2 else np.array([0.0, 0.0], dtype=float)

        for step in range(1, int(horizon) + 1):
            pred = last + slope * step
            pred_date = current_date + pd.Timedelta(days=step)
            rows.append(
                {
                    "run_id": run_id,
                    "transect_id": int(transect_id),
                    "datetime": pred_date.isoformat(),
                    "future_VE_distance_m": float(pred[0]),
                    "future_WL_distance_m": float(pred[1]),
                    "model_type": model_type,
                }
            )
    return pd.DataFrame(rows)


def run_prediction(
    run_id: str,
    train_split_date: str | None = None,
    sequence_len_days: int = 10,
    forecast_days: int = 30,
    model_preference: str = "mamba_lstm",
    allow_training: bool = True,
    warm_start_model_path: str | None = None,
) -> PredictionArtifacts:
    """Run VE/WL prediction in train+forecast mode or fast inference-only mode."""
    run_dir = Path("data") / "runs" / run_id
    results_dir = run_dir / "results"
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = results_dir / "timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(f"Timeseries file not found for run {run_id}")

    raw = pd.read_csv(timeseries_path)
    if raw.empty:
        raise ValueError("Timeseries is empty")

    raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True)
    transect_ids = sorted(raw["transect_id"].dropna().astype(int).unique().tolist())

    seq_len = max(2, int(sequence_len_days))
    horizon = max(1, int(forecast_days))

    daily_map: dict[int, pd.DataFrame] = {}
    daily_map_all: dict[int, pd.DataFrame] = {}
    x_all = []
    y_all = []
    sample_dates = []

    for transect_id in transect_ids:
        daily = _build_daily_series(raw[raw["transect_id"] == transect_id])
        if daily.empty:
            continue
        daily_map_all[transect_id] = daily

        if len(daily) <= seq_len + 2:
            continue
        daily_map[transect_id] = daily

        values = daily[["VE_distance_m", "WL_distance_m"]].to_numpy(dtype=float)
        x, y = _create_sequences(values, seq_len)
        if len(x) == 0:
            continue

        target_dates = daily.index[seq_len:]
        x_all.append(x)
        y_all.append(y)
        sample_dates.extend(target_dates)

    if not daily_map_all:
        raise ValueError("Insufficient data after interpolation for sequence modeling")

    if not x_all:
        model_type = "persistence_fallback"
        training_info: dict[str, Any] = {
            "allow_training": bool(allow_training),
            "training_skipped": True,
            "warning": "Insufficient sequence samples for model inference. Used sparse fallback forecast.",
        }
        rmse_ve = np.nan
        rmse_wl = np.nan
        metrics_df = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "model_type": model_type,
                    "rmse_ve_m": rmse_ve,
                    "rmse_wl_m": rmse_wl,
                    "train_samples": 0,
                    "val_samples": 0,
                }
            ]
        )

        forecast_df = _build_sparse_fallback_forecast(
            run_id=run_id,
            daily_map=daily_map_all,
            horizon=horizon,
            model_type=model_type,
        )
        if forecast_df.empty:
            raise ValueError("Insufficient data after interpolation for sequence modeling")

        summary = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model_type": model_type,
            "model_preference": model_preference,
            "allow_training": bool(allow_training),
            "sequence_len_days": seq_len,
            "forecast_days": horizon,
            "forecast_years": round(float(horizon) / 365.25, 3),
            "transect_count": len(daily_map_all),
            "modeled_transect_count": 0,
            "rmse_ve_m": rmse_ve,
            "rmse_wl_m": rmse_wl,
            "training": training_info,
        }

        metrics_df.to_csv(pred_dir / "metrics.csv", index=False)
        forecast_df.to_csv(pred_dir / "forecast.csv", index=False)
        try:
            forecast_df.to_parquet(pred_dir / "forecast.parquet", index=False)
        except Exception:
            pass

        run_summary_path = run_dir / "summary.json"
        utm_epsg = 4326
        if run_summary_path.exists():
            with run_summary_path.open("r", encoding="utf-8") as f:
                run_summary = json.load(f)
                utm_epsg = int(run_summary.get("utm_epsg", 4326))

        forecast_shoreline_geojson = _build_forecast_shoreline_geojson(
            run_id=run_id,
            forecast_df=forecast_df,
            transects_path=results_dir / "transects.geojson",
            utm_epsg=utm_epsg,
        )
        with (pred_dir / "shoreline_forecast.geojson").open("w", encoding="utf-8") as f:
            json.dump(forecast_shoreline_geojson, f, indent=2)

        summary["files"] = {
            "forecast_csv": str(pred_dir / "forecast.csv"),
            "metrics_csv": str(pred_dir / "metrics.csv"),
            "forecast_geojson": str(pred_dir / "shoreline_forecast.geojson"),
        }

        with (pred_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return PredictionArtifacts(summary=summary, forecast_df=forecast_df, metrics_df=metrics_df)

    x_full = np.concatenate(x_all, axis=0)
    y_full = np.concatenate(y_all, axis=0)
    sample_dates = pd.to_datetime(sample_dates, utc=True)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_shape = x_full.shape
    x_scaled = scaler_x.fit_transform(x_full.reshape(-1, x_shape[2])).reshape(x_shape)
    y_scaled = scaler_y.fit_transform(y_full)

    if train_split_date:
        split_dt = pd.Timestamp(train_split_date, tz="UTC")
        train_mask = sample_dates <= split_dt
        if not train_mask.any() or train_mask.all():
            split_idx = int(len(x_scaled) * 0.8)
            train_mask = np.array([idx < split_idx for idx in range(len(x_scaled))])
    else:
        split_idx = int(len(x_scaled) * 0.8)
        train_mask = np.array([idx < split_idx for idx in range(len(x_scaled))])

    x_train = x_scaled[train_mask]
    y_train = y_scaled[train_mask]
    x_val = x_scaled[~train_mask]
    y_val = y_scaled[~train_mask]

    model_type = "persistence_fallback"
    training_info: dict[str, Any] = {"allow_training": bool(allow_training)}
    mamba_model = None
    tf_model = None
    backend_errors: list[str] = []

    wants_mamba = str(model_preference).lower() in {"mamba", "mamba_lstm", "auto"}
    if allow_training:
        if wants_mamba:
            try:
                mamba_model, training_info = _train_mamba_lstm_torch(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    pred_dir=pred_dir,
                    warm_start_model_path=warm_start_model_path,
                )
                model_type = "mamba_lstm"
            except Exception as exc:
                backend_errors.append(f"mamba_lstm: {exc}")

        if mamba_model is None:
            try:
                import tensorflow  # noqa: F401

                if len(x_train) < 8 or len(x_val) < 2:
                    raise RuntimeError("Not enough samples for stable LSTM training.")
                tf_model, training_info = _train_tf_lstm(x_train, y_train, x_val, y_val)
                tf_model.save(pred_dir / "lstm_model.keras")
                model_type = "lstm"
            except Exception as exc:
                backend_errors.append(f"lstm: {exc}")
                training_info["warning"] = "Falling back to persistence baseline."
    else:
        mamba_model, load_info = _load_mamba_lstm_model(warm_start_model_path)
        if mamba_model is not None:
            model_type = "mamba_lstm_pretrained"
            training_info.update(load_info)
            training_info["training_skipped"] = True
        else:
            training_info.update(load_info)
            training_info["training_skipped"] = True

    if backend_errors:
        training_info["backend_errors"] = backend_errors
    training_info["allow_training"] = bool(allow_training)
    if warm_start_model_path:
        training_info["warm_start_model_path"] = str(warm_start_model_path)

    if len(x_val) > 0:
        if mamba_model is not None:
            y_pred_scaled = _predict_mamba_torch(mamba_model, x_val)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_val)
        elif tf_model is not None:
            y_pred_scaled = tf_model.predict(x_val, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_val)
        else:
            y_pred = scaler_y.inverse_transform(x_val[:, -1, :])
            y_true = scaler_y.inverse_transform(y_val)
    else:
        y_pred = np.empty((0, 2))
        y_true = np.empty((0, 2))

    rmse_ve = float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))) if len(y_true) else np.nan
    rmse_wl = float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))) if len(y_true) else np.nan

    metrics_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "model_type": model_type,
                "rmse_ve_m": rmse_ve,
                "rmse_wl_m": rmse_wl,
                "train_samples": int(len(x_train)),
                "val_samples": int(len(x_val)),
            }
        ]
    )

    forecast_rows: list[dict[str, Any]] = []
    for transect_id, daily in daily_map.items():
        window = daily[["VE_distance_m", "WL_distance_m"]].to_numpy(dtype=float)
        if len(window) < seq_len:
            continue

        seq_raw = window[-seq_len:, :]
        seq_scaled = scaler_x.transform(seq_raw).reshape(1, seq_len, 2)
        current_date = daily.index[-1]

        for step in range(horizon):
            if mamba_model is not None:
                pred_scaled = _predict_mamba_torch(mamba_model, seq_scaled)
                pred = scaler_y.inverse_transform(pred_scaled)[0]
            elif tf_model is not None:
                pred_scaled = tf_model.predict(seq_scaled, verbose=0)
                pred = scaler_y.inverse_transform(pred_scaled)[0]
            else:
                pred = seq_raw[-1, :]

            pred_date = current_date + pd.Timedelta(days=step + 1)
            forecast_rows.append(
                {
                    "run_id": run_id,
                    "transect_id": transect_id,
                    "datetime": pred_date.isoformat(),
                    "future_VE_distance_m": float(pred[0]),
                    "future_WL_distance_m": float(pred[1]),
                    "model_type": model_type,
                }
            )

            next_scaled = scaler_x.transform(np.array(pred).reshape(1, -1))
            seq_scaled = np.concatenate([seq_scaled[:, 1:, :], next_scaled.reshape(1, 1, 2)], axis=1)
            seq_raw = np.concatenate([seq_raw[1:, :], np.array(pred).reshape(1, 2)], axis=0)

    forecast_df = pd.DataFrame(forecast_rows)

    summary = {
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_type": model_type,
        "model_preference": model_preference,
        "allow_training": bool(allow_training),
        "sequence_len_days": seq_len,
        "forecast_days": horizon,
        "forecast_years": round(float(horizon) / 365.25, 3),
        "transect_count": len(daily_map_all),
        "modeled_transect_count": len(daily_map),
        "rmse_ve_m": rmse_ve,
        "rmse_wl_m": rmse_wl,
        "training": training_info,
    }

    metrics_df.to_csv(pred_dir / "metrics.csv", index=False)
    forecast_df.to_csv(pred_dir / "forecast.csv", index=False)
    try:
        forecast_df.to_parquet(pred_dir / "forecast.parquet", index=False)
    except Exception:
        pass

    run_summary_path = run_dir / "summary.json"
    utm_epsg = 4326
    if run_summary_path.exists():
        with run_summary_path.open("r", encoding="utf-8") as f:
            run_summary = json.load(f)
            utm_epsg = int(run_summary.get("utm_epsg", 4326))

    forecast_shoreline_geojson = _build_forecast_shoreline_geojson(
        run_id=run_id,
        forecast_df=forecast_df,
        transects_path=results_dir / "transects.geojson",
        utm_epsg=utm_epsg,
    )
    with (pred_dir / "shoreline_forecast.geojson").open("w", encoding="utf-8") as f:
        json.dump(forecast_shoreline_geojson, f, indent=2)

    summary["files"] = {
        "forecast_csv": str(pred_dir / "forecast.csv"),
        "metrics_csv": str(pred_dir / "metrics.csv"),
        "forecast_geojson": str(pred_dir / "shoreline_forecast.geojson"),
    }

    with (pred_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return PredictionArtifacts(summary=summary, forecast_df=forecast_df, metrics_df=metrics_df)
