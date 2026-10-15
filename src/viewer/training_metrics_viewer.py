from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
MAX_PLOT_Y = 500.0
SURFACE_THICKNESS_FIGURE_HEIGHT = 1100
SURFACE_THICKNESS_ANIMATION_HEIGHT = 1300


@dataclass(frozen=True)
class TrainingRunRecord:
    """A training run discovered from an experiment JSON log."""

    experiment_family: str
    experiment_id: str
    log_path: Path
    updated_at: float


@dataclass(frozen=True)
class SurfaceThicknessRollout:
    """Surface-thickness rollout data loaded from a NetCDF file."""

    x: np.ndarray
    y: np.ndarray
    time_days: np.ndarray
    truth: np.ndarray
    rollout: np.ndarray
    zmin: float
    zmax: float


def _resolve_surface_thickness_arrays(dataset: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return truth and rollout surface-thickness arrays from 1-layer or 2-layer rollouts."""

    truth_name = "truth_surface_thickness" if "truth_surface_thickness" in dataset.data_vars else "truth_layer_thickness"
    rollout_name = "rollout_surface_thickness" if "rollout_surface_thickness" in dataset.data_vars else "rollout_layer_thickness"
    truth = dataset[truth_name]
    rollout = dataset[rollout_name]
    if "layers" in truth.dims:
        truth = truth.isel(layers=0)
    if "layers" in rollout.dims:
        rollout = rollout.isel(layers=0)
    return truth, rollout


def log_updated_at(payload: dict[str, Any], fallback: float) -> float:
    """Return the true log update/completion timestamp, falling back to file mtime."""

    value = payload.get("last_update_time")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return fallback
    return fallback


def list_training_runs(data_root: Path = DATA_ROOT) -> list[TrainingRunRecord]:
    """List current and completed jobs from ``data/<family>/logs/*.json``."""

    records: list[TrainingRunRecord] = []
    for log_path in sorted(data_root.glob("*/logs/*.json")):
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        experiment_id = str(metadata.get("experiment_id", log_path.stem))
        records.append(
            TrainingRunRecord(
                experiment_family=log_path.parents[1].name,
                experiment_id=experiment_id,
                log_path=log_path,
                updated_at=log_updated_at(payload, log_path.stat().st_mtime),
            )
        )
    return sorted(records, key=lambda record: (record.updated_at, record.experiment_id), reverse=True)


def load_experiment_log(record_or_path: TrainingRunRecord | Path) -> dict[str, Any]:
    """Load an experiment JSON log."""

    log_path = getattr(record_or_path, "log_path", None)
    path = Path(log_path) if log_path is not None else Path(record_or_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def format_updated_at(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def run_status(log: dict[str, Any]) -> str:
    """Infer a lightweight status for a run from its epoch count."""

    metadata = log.get("metadata", {}) if isinstance(log.get("metadata"), dict) else {}
    requested = int(metadata.get("epochs_requested", 0) or 0)
    completed = len(log.get("epochs", []) if isinstance(log.get("epochs"), list) else [])
    if requested and completed >= requested:
        return "completed"
    if completed > 0:
        return "running_or_interrupted"
    return "created"


def latest_rollout_artifact_path(log: dict[str, Any]) -> Path | None:
    rollouts = log.get("rollouts", [])
    if not isinstance(rollouts, list) or not rollouts:
        return None
    for rollout in reversed(rollouts):
        if not isinstance(rollout, dict):
            continue
        artifact = rollout.get("rollout_artifact")
        if isinstance(artifact, dict) and artifact.get("saved") and artifact.get("path"):
            path = Path(str(artifact["path"]))
            return path if path.exists() else path
        if rollout.get("rollout_path"):
            path = Path(str(rollout["rollout_path"]))
            return path if path.exists() else path
        if rollout.get("latest_surface_thickness_path"):
            path = Path(str(rollout["latest_surface_thickness_path"]))
            return path if path.exists() else path
    return None


def latest_rollout_path(log: dict[str, Any]) -> Path | None:
    """Backward-compatible alias for the latest available rollout artifact."""

    return latest_rollout_artifact_path(log)


def summarize_runs(records: list[TrainingRunRecord]) -> list[dict[str, Any]]:
    """Build table rows for the Streamlit overview."""

    rows: list[dict[str, Any]] = []
    for record in records:
        log = load_experiment_log(record)
        metadata = log.get("metadata", {}) if isinstance(log.get("metadata"), dict) else {}
        epochs = log.get("epochs", []) if isinstance(log.get("epochs"), list) else []
        rollouts = log.get("rollouts", []) if isinstance(log.get("rollouts"), list) else []
        last_rollout = rollouts[-1] if rollouts and isinstance(rollouts[-1], dict) else {}
        rows.append(
            {
                "updated_at": format_updated_at(record.updated_at),
                "family": record.experiment_family,
                "experiment_id": record.experiment_id,
                "hypothesis": metadata.get("hypothesis"),
                "final_eval_mse": float(np.mean(rollout_mse_series(last_rollout))) if last_rollout else None,
                "status": run_status(log),
                "device": metadata.get("device"),
                "epochs_completed": len(epochs),
                "epochs_requested": metadata.get("epochs_requested"),
                "latest_eval_mse_last": last_rollout.get("eval_mse_last"),
                "latest_eval_mse_timesteps": len(last_rollout.get("eval_mse_by_time", [])) if isinstance(last_rollout.get("eval_mse_by_time"), list) else None,
                "log_path": str(record.log_path),
            }
        )
    return rows


def records_by_family(records: list[TrainingRunRecord]) -> dict[str, list[TrainingRunRecord]]:
    """Group records by experiment family, preserving most-recent-first order."""

    grouped: dict[str, list[TrainingRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.experiment_family, []).append(record)
    return grouped


def training_loss_figure(log: dict[str, Any]) -> go.Figure:
    epochs = log.get("epochs", []) if isinstance(log.get("epochs"), list) else []
    x = [int(row["epoch"]) for row in epochs if isinstance(row, dict) and row.get("epoch") is not None]
    y = [float(row["training_loss"]) for row in epochs if isinstance(row, dict) and row.get("training_loss") is not None]
    figure = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines+markers", name="training_loss", marker={"symbol": "circle"})])
    figure.update_layout(title="Per-epoch training loss", xaxis_title="epoch", yaxis_title="loss", yaxis={"rangemode": "tozero"})
    return figure


def epoch_timing_figure(log: dict[str, Any]) -> go.Figure:
    epochs = log.get("epochs", []) if isinstance(log.get("epochs"), list) else []
    x = [int(row["epoch"]) for row in epochs if isinstance(row, dict) and row.get("epoch") is not None]
    y = [float(row["duration_seconds"]) for row in epochs if isinstance(row, dict) and row.get("duration_seconds") is not None]
    figure = go.Figure(data=[go.Bar(x=x, y=y, name="duration_seconds")])
    figure.update_layout(title="Per-epoch timing", xaxis_title="epoch", yaxis_title="seconds", yaxis={"rangemode": "tozero"})
    return figure


def rollout_mse_by_time_from_netcdf(rollout_path: Path) -> list[float]:
    """Compute per-timestep MSE from a saved rollout NetCDF file."""

    with xr.open_dataset(rollout_path) as dataset:
        truth, rollout = _resolve_surface_thickness_arrays(dataset)
        spatial_dims = [dim for dim in truth.dims if dim != "time_days"]
        series = ((truth - rollout) ** 2).mean(dim=spatial_dims).values
    return [float(value) for value in np.asarray(series).reshape(-1)]


def rollout_mse_series(rollout: dict[str, Any]) -> list[float]:
    """Return per-timestep rollout MSE from the log or compute it from NetCDF."""

    series = rollout.get("eval_mse_by_time")
    if isinstance(series, list):
        return [float(value) for value in series]
    path_value = rollout.get("rollout_path")
    if path_value is None:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return []
    return rollout_mse_by_time_from_netcdf(path)


def _rms_by_time(array: xr.DataArray) -> list[float]:
    """Compute per-timestep spatial RMS for a surface-thickness array."""

    spatial_dims = [dim for dim in array.dims if dim != "time_days"]
    # Rollout NetCDF files use NaN fill values for masked cells; skip them so
    # each timestep RMS is computed over finite surface points only.
    series = np.sqrt((array**2).mean(dim=spatial_dims, skipna=True).values)
    return [float(value) for value in np.asarray(series).reshape(-1)]


def rollout_surface_thickness_rms_by_time_from_netcdf(rollout_path: Path) -> list[float]:
    """Compute per-timestep rollout surface-thickness RMS from a saved rollout NetCDF file."""

    with xr.open_dataset(rollout_path) as dataset:
        _, rollout = _resolve_surface_thickness_arrays(dataset)
        return _rms_by_time(rollout)


def truth_surface_thickness_rms_by_time_from_netcdf(rollout_path: Path) -> list[float]:
    """Compute per-timestep truth surface-thickness RMS from a saved rollout NetCDF file."""

    with xr.open_dataset(rollout_path) as dataset:
        truth, _ = _resolve_surface_thickness_arrays(dataset)
        return _rms_by_time(truth)


def surface_thickness_rms_series(rollout: dict[str, Any]) -> list[float]:
    """Return per-timestep rollout surface-thickness RMS from the rollout NetCDF."""

    path_value = rollout.get("rollout_path")
    if path_value is None:
        return []
    path = Path(str(path_value))
    if not path.exists():
        return []
    return rollout_surface_thickness_rms_by_time_from_netcdf(path)


def rollout_metrics_figure(log: dict[str, Any]) -> go.Figure:
    rollouts = log.get("rollouts", []) if isinstance(log.get("rollouts"), list) else []
    figure = go.Figure()
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        epoch = rollout.get("epoch")
        series = rollout_mse_series(rollout)
        if epoch is None or not series:
            continue
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(series) + 1)),
                y=series,
                mode="lines+markers",
                marker={"symbol": "circle"},
                name=f"epoch {epoch}",
            )
        )
    figure.update_layout(title="Rollout MSE by eval timestep", xaxis_title="eval timestep", yaxis_title="MSE", yaxis={"range": [0, MAX_PLOT_Y]})
    return figure


def surface_thickness_rms_figure(log: dict[str, Any]) -> go.Figure:
    """Plot rollout and truth surface-thickness RMS by eval timestep."""

    rollouts = log.get("rollouts", []) if isinstance(log.get("rollouts"), list) else []
    figure = go.Figure()
    truth_series: list[float] = []
    for rollout in rollouts:
        if not isinstance(rollout, dict):
            continue
        epoch = rollout.get("epoch")
        series = surface_thickness_rms_series(rollout)
        if epoch is None or not series:
            continue
        if not truth_series:
            path_value = rollout.get("rollout_path")
            if path_value is not None:
                path = Path(str(path_value))
                if path.exists():
                    truth_series = truth_surface_thickness_rms_by_time_from_netcdf(path)
        if not np.isfinite(series).any():
            continue
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(series) + 1)),
                y=series,
                mode="lines+markers",
                marker={"symbol": "circle"},
                name=f"epoch {epoch} rollout RMS",
            )
        )
    if truth_series and np.isfinite(truth_series).any():
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(truth_series) + 1)),
                y=truth_series,
                mode="lines+markers",
                line={"color": "white", "dash": "dash"},
                marker={"symbol": "x", "color": "white"},
                name="truth RMS",
            )
        )
    figure.update_layout(title="Surface thickness RMS by eval timestep", xaxis_title="eval timestep", yaxis_title="RMS", yaxis={"rangemode": "tozero"})
    return figure


def overview_eval_mean_mse_figure(records: list[TrainingRunRecord]) -> go.Figure:
    """Scatter final eval MSE against experiment index."""

    sorted_records = sorted(records, key=lambda item: (item.updated_at, item.experiment_id))
    x_values: list[int] = []
    y_values: list[float] = []
    hover_text: list[str] = []
    labels: list[str] = []
    marker_colours: list[str] = []
    best_so_far = float("inf")
    for index, record in enumerate(sorted_records, start=1):
        log = load_experiment_log(record)
        rollouts = log.get("rollouts", []) if isinstance(log.get("rollouts"), list) else []
        final_series: list[float] = []
        for rollout in reversed(rollouts):
            if isinstance(rollout, dict):
                final_series = rollout_mse_series(rollout)
                if final_series:
                    break
        if not final_series:
            continue
        final_eval_mse = float(np.mean(final_series))
        is_new_best = final_eval_mse < best_so_far
        if is_new_best:
            best_so_far = final_eval_mse
        x_values.append(index)
        y_values.append(final_eval_mse)
        labels.append(record.experiment_id)
        marker_colours.append("red" if is_new_best else "#2563eb")
        hover_text.append(
            f"index: {index}<br>family: {record.experiment_family}<br>experiment: {record.experiment_id}"
            f"<br>final eval MSE: {final_eval_mse:.6g}<br>new best incumbent: {is_new_best}"
        )

    figure = go.Figure(
        data=[
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                text=labels,
                hovertext=hover_text,
                hoverinfo="text",
                marker={"size": 11, "color": marker_colours},
                name="final_eval_mse",
            )
        ]
    )
    figure.update_layout(
        title="Final eval MSE by experiment index",
        xaxis_title="experiment index",
        yaxis_title="final eval MSE",
        yaxis={"range": [0, MAX_PLOT_Y]},
    )
    return figure


def load_surface_thickness_rollout(rollout_path: Path) -> SurfaceThicknessRollout:
    """Load truth and rollout surface-thickness series from a rollout NetCDF."""

    with xr.open_dataset(rollout_path) as dataset:
        truth, rollout = _resolve_surface_thickness_arrays(dataset)
        if "time_days" not in truth.dims:
            truth = truth.expand_dims(time_days=[float(np.asarray(dataset.coords["time_days"].values).reshape(-1)[0])])
        if "time_days" not in rollout.dims:
            rollout = rollout.expand_dims(time_days=[float(np.asarray(dataset.coords["time_days"].values).reshape(-1)[0])])
        x = truth.coords["x"].values
        y = truth.coords["y"].values
        zmin = float(min(truth.min().item(), rollout.min().item()))
        zmax = float(max(truth.max().item(), rollout.max().item()))
        time_days = np.asarray(truth.coords["time_days"].values, dtype=float).reshape(-1)
        truth_values = np.asarray(truth.values)
        rollout_values = np.asarray(rollout.values)

    return SurfaceThicknessRollout(
        x=np.asarray(x),
        y=np.asarray(y),
        time_days=time_days,
        truth=truth_values,
        rollout=rollout_values,
        zmin=zmin,
        zmax=zmax,
    )


def surface_thickness_frame_figure(data: SurfaceThicknessRollout, time_index: int) -> go.Figure:
    """Plot truth and rollout surface-thickness fields for a single timestep."""

    time_value = float(data.time_days[time_index])

    figure = make_subplots(rows=1, cols=2, subplot_titles=("Truth", "Rollout"), horizontal_spacing=0.08)
    figure.add_trace(
        go.Heatmap(x=data.x, y=data.y, z=data.truth[time_index], zmin=data.zmin, zmax=data.zmax, coloraxis="coloraxis"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(x=data.x, y=data.y, z=data.rollout[time_index], zmin=data.zmin, zmax=data.zmax, coloraxis="coloraxis"),
        row=1,
        col=2,
    )
    figure.update_layout(
        title=f"Surface thickness truth vs rollout at day {time_value:.4f}",
        coloraxis={"colorscale": "RdBu_r", "cmin": data.zmin, "cmax": data.zmax},
        height=SURFACE_THICKNESS_FIGURE_HEIGHT,
        margin={"l": 40, "r": 40, "t": 90, "b": 40},
    )
    figure.update_xaxes(title_text="x", constrain="domain")
    figure.update_yaxes(title_text="y", scaleanchor="x", scaleratio=1)
    figure.update_xaxes(title_text="x", constrain="domain", row=1, col=2)
    figure.update_yaxes(title_text="y", scaleanchor="x2", scaleratio=1, row=1, col=2)
    return figure


def surface_thickness_animation_figure(
    data: SurfaceThicknessRollout,
    *,
    frame_duration_ms: int = 233,
) -> go.Figure:
    """Build a single Plotly figure with client-side animation frames."""

    figure = surface_thickness_frame_figure(data, 0)
    figure.frames = [
        go.Frame(
            name=f"timestep-{time_index}",
            data=[
                go.Heatmap(z=data.truth[time_index], x=data.x, y=data.y, zmin=data.zmin, zmax=data.zmax, coloraxis="coloraxis"),
                go.Heatmap(z=data.rollout[time_index], x=data.x, y=data.y, zmin=data.zmin, zmax=data.zmax, coloraxis="coloraxis"),
            ],
            traces=[0, 1],
            layout=go.Layout(title=f"Surface thickness truth vs rollout at day {float(data.time_days[time_index]):.4f}"),
        )
        for time_index in range(data.truth.shape[0])
    ]
    frame_names = [frame.name for frame in figure.frames]
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            frame_names,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": -0.08,
                "len": 0.84,
                "currentvalue": {"prefix": "Frame: "},
                "steps": [
                    {
                        "label": f"{float(data.time_days[time_index]):.4f}",
                        "method": "animate",
                        "args": [
                            [f"timestep-{time_index}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for time_index in range(data.truth.shape[0])
                ],
            }
        ],
    )
    return figure


def final_rollout_heatmap_figure(rollout_path: Path) -> go.Figure:
    """Plot the final truth and rollout surface-thickness fields from NetCDF."""

    data = load_surface_thickness_rollout(rollout_path)
    return surface_thickness_frame_figure(data, -1)


def play_surface_thickness_animation(  # pragma: no cover - Streamlit UI wrapper
    st_module: Any,
    rollout_path: Path,
    *,
    frame_duration_ms: int = 233,
) -> None:
    """Render a client-side animation of the latest surface-thickness rollout."""

    data = load_surface_thickness_rollout(rollout_path)
    if data.truth.shape[0] == 0:
        st_module.info("The rollout file does not contain any surface-thickness timesteps.")
        return

    figure = surface_thickness_animation_figure(data, frame_duration_ms=frame_duration_ms)
    html = figure.to_html(include_plotlyjs=True, full_html=False)
    autoplay_script = """
<script>
const plot = document.querySelector('.plotly-graph-div');
if (plot && window.Plotly) {
  window.Plotly.animate(plot, null, {
    frame: {duration: FRAME_MS, redraw: true},
    transition: {duration: 0},
    mode: 'immediate'
  });
}
</script>
""".replace("FRAME_MS", str(frame_duration_ms))
    st_module.components.v1.html(html + autoplay_script, height=SURFACE_THICKNESS_ANIMATION_HEIGHT, scrolling=False)


def render_training_metrics_page(st_module=st, data_root: Path = DATA_ROOT) -> None:  # pragma: no cover - Streamlit UI wrapper
    """Render the Streamlit app for current and future experiment jobs."""

    st_module.set_page_config(page_title="Ocean Emulator Training", layout="wide")
    st_module.title("Ocean Emulator Training Progress")
    records = list_training_runs(data_root)
    if not records:
        st_module.warning(f"No experiment logs found under `{data_root}/*/logs/*.json`.")
        st_module.stop()

    grouped_records = records_by_family(records)
    families = list(grouped_records)
    selected_family = st_module.sidebar.radio(
        "Experiment family",
        families,
    )
    family_records = grouped_records[selected_family]
    detail_tab, overview_tab = st_module.tabs(["Run detail", "Overview"])
    with detail_tab:
        selected = st_module.sidebar.selectbox(
            "Experiment",
            family_records,
            format_func=lambda record: f"{record.experiment_id} ({format_updated_at(record.updated_at)})",
        )
        log = load_experiment_log(selected)
        metadata = log.get("metadata", {}) if isinstance(log.get("metadata"), dict) else {}
        st_module.caption(
            f"Status `{run_status(log)}` | device `{metadata.get('device', 'unknown')}` | "
            f"last update `{log.get('last_update_time', format_updated_at(selected.updated_at))}`"
        )
        col1, col2, col3 = st_module.columns(3)
        col1.metric("Epochs completed", len(log.get("epochs", []) if isinstance(log.get("epochs"), list) else []))
        col2.metric("Epochs requested", metadata.get("epochs_requested", "unknown"))
        col3.metric("Rollouts", len(log.get("rollouts", []) if isinstance(log.get("rollouts"), list) else []))

        chart_col1, chart_col2 = st_module.columns(2)
        chart_col1.plotly_chart(epoch_timing_figure(log), width="stretch")
        chart_col2.plotly_chart(rollout_metrics_figure(log), width="stretch")
        rollout_col1, rollout_col2 = st_module.columns(2)
        rollout_col1.plotly_chart(training_loss_figure(log), width="stretch")
        rollout_col2.plotly_chart(surface_thickness_rms_figure(log), width="stretch")

        rollout_path = latest_rollout_artifact_path(log)
        st_module.subheader("Latest surface artifact")
        if rollout_path is None:
            st_module.info("No surface `.nc` artifact is saved for this run.")
        else:
            st_module.caption(f"`{rollout_path}`")
            if rollout_path.exists():
                artifact_data = load_surface_thickness_rollout(rollout_path)
                st_module.plotly_chart(surface_thickness_frame_figure(artifact_data, -1), width="stretch")
                if len(artifact_data.time_days) > 1:
                    st_module.caption("Surface thickness rollout animation")
                    play_surface_thickness_animation(
                        st_module,
                        rollout_path,
                    )
            else:
                st_module.warning("The latest surface artifact path is logged but does not exist on disk.")

        st_module.subheader("Experiment log")
        st_module.json(log)

    with overview_tab:
        st_module.subheader(f"Current and recent jobs for `{selected_family}`")
        st_module.plotly_chart(overview_eval_mean_mse_figure(family_records), width="stretch")
        st_module.dataframe(summarize_runs(family_records), use_container_width=True)


def main(st_module=st) -> None:  # pragma: no cover - Streamlit UI wrapper
    render_training_metrics_page(st_module)


if __name__ == "__main__":  # pragma: no cover
    main()
