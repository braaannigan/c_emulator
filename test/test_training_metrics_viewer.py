import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.viewer.training_metrics_viewer import (
    TrainingRunRecord,
    final_rollout_heatmap_figure,
    format_updated_at,
    load_surface_thickness_rollout,
    latest_rollout_path,
    list_training_runs,
    log_updated_at,
    load_experiment_log,
    epoch_timing_figure,
    MAX_PLOT_Y,
    overview_eval_mean_mse_figure,
    records_by_family,
    rollout_metrics_figure,
    rollout_mse_by_time_from_netcdf,
    rollout_mse_series,
    rollout_surface_thickness_rms_by_time_from_netcdf,
    surface_thickness_rms_figure,
    surface_thickness_rms_series,
    truth_surface_thickness_rms_by_time_from_netcdf,
    run_status,
    surface_thickness_animation_figure,
    surface_thickness_frame_figure,
    summarize_runs,
    training_loss_figure,
)


def write_log(path, experiment_id="exp", epochs=None, rollouts=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"experiment_id": experiment_id, "epochs_requested": 2, "device": "cpu"},
        "epochs": epochs if epochs is not None else [{"epoch": 1, "training_loss": 3.0, "duration_seconds": 0.5}],
        "rollouts": rollouts if rollouts is not None else [{"epoch": 1, "eval_mse_by_time": [4.0, 5.0], "eval_mse_last": 5.0, "rollout_path": "rollout.nc"}],
        "last_update_time": "2026-01-01T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_log_updated_at_prefers_json_timestamp_over_file_mtime():
    assert format_updated_at(log_updated_at({"last_update_time": "2026-05-04T10:17:22+00:00"}, 1.0)).startswith("2026-05-04T10:17:22")
    assert log_updated_at({"last_update_time": "bad"}, 123.0) == 123.0


def test_list_training_runs_discovers_json_logs(tmp_path):
    log_path = tmp_path / "double_gyre_shifting_wind" / "logs" / "exp.json"
    write_log(log_path, experiment_id="exp")
    (tmp_path / "bad" / "logs").mkdir(parents=True)
    (tmp_path / "bad" / "logs" / "bad.json").write_text("not-json", encoding="utf-8")

    records = list_training_runs(tmp_path)

    assert len(records) == 1
    assert records[0].experiment_family == "double_gyre_shifting_wind"
    assert records[0].experiment_id == "exp"


def test_list_training_runs_sorts_by_logged_update_time_not_file_mtime(tmp_path):
    old_path = tmp_path / "family" / "logs" / "old.json"
    new_path = tmp_path / "family" / "logs" / "new.json"
    write_log(old_path, experiment_id="old")
    write_log(new_path, experiment_id="new")
    old_payload = json.loads(old_path.read_text(encoding="utf-8"))
    old_payload["last_update_time"] = "2026-05-04T09:00:00+00:00"
    old_path.write_text(json.dumps(old_payload), encoding="utf-8")
    new_payload = json.loads(new_path.read_text(encoding="utf-8"))
    new_payload["last_update_time"] = "2026-05-04T10:00:00+00:00"
    new_path.write_text(json.dumps(new_payload), encoding="utf-8")
    old_path.touch()

    records = list_training_runs(tmp_path)

    assert [record.experiment_id for record in records] == ["new", "old"]


def test_records_by_family_groups_runs_in_existing_recency_order():
    records = [
        TrainingRunRecord("family_b", "exp3", Path("b3.json"), 30.0),
        TrainingRunRecord("family_a", "exp2", Path("a2.json"), 20.0),
        TrainingRunRecord("family_b", "exp1", Path("b1.json"), 10.0),
    ]

    grouped = records_by_family(records)

    assert list(grouped) == ["family_b", "family_a"]
    assert [record.experiment_id for record in grouped["family_b"]] == ["exp3", "exp1"]
    assert [record.experiment_id for record in grouped["family_a"]] == ["exp2"]


def test_records_by_family_groups_records():
    records = [
        TrainingRunRecord("a", "one", Path("one.json"), 1.0),
        TrainingRunRecord("b", "two", Path("two.json"), 2.0),
        TrainingRunRecord("a", "three", Path("three.json"), 3.0),
    ]

    grouped = records_by_family(records)

    assert [record.experiment_id for record in grouped["a"]] == ["one", "three"]
    assert [record.experiment_id for record in grouped["b"]] == ["two"]


def test_load_status_summary_and_figures(tmp_path):
    log_path = tmp_path / "family" / "logs" / "exp.json"
    write_log(log_path)
    record = TrainingRunRecord("family", "exp", log_path, log_path.stat().st_mtime)

    log = load_experiment_log(record)
    rows = summarize_runs([record])

    assert run_status(log) == "running_or_interrupted"
    assert "final_eval_mse" in rows[0]
    assert rows[0]["final_eval_mse"] == 4.5
    assert rows[0]["latest_eval_mse_last"] == 5.0
    assert rows[0]["latest_eval_mse_timesteps"] == 2
    assert len(training_loss_figure(log).data) == 1
    assert training_loss_figure(log).data[0].marker.symbol == "circle"
    assert training_loss_figure(log).layout.yaxis.range is None
    assert len(epoch_timing_figure(log).data) == 1
    assert epoch_timing_figure(log).layout.yaxis.range is None
    assert len(rollout_metrics_figure(log).data) == 1
    assert rollout_metrics_figure(log).data[0].marker.symbol == "circle"
    assert tuple(rollout_metrics_figure(log).layout.yaxis.range) == (0, 500.0)


def test_load_experiment_log_accepts_record_like_objects(tmp_path):
    log_path = tmp_path / "family" / "logs" / "exp.json"
    payload = write_log(log_path)

    class LegacyTrainingRunRecord:
        def __init__(self, path):
            self.log_path = path

    assert load_experiment_log(LegacyTrainingRunRecord(log_path)) == payload


def test_run_status_completed_and_created():
    assert run_status({"metadata": {"epochs_requested": 1}, "epochs": [{"epoch": 1}]}) == "completed"
    assert run_status({"metadata": {"epochs_requested": 1}, "epochs": []}) == "created"


def test_latest_rollout_path_returns_none_when_no_rollout_is_logged():
    assert latest_rollout_path({"rollouts": []}) is None
    assert latest_rollout_path({"rollouts": ["not-a-dict"]}) is None


def test_latest_rollout_path_returns_last_logged_path(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    rollout_path.write_text("placeholder", encoding="utf-8")

    path = latest_rollout_path({"rollouts": [{"rollout_path": "old.nc"}, {"rollout_path": str(rollout_path)}]})

    assert path == rollout_path


def test_rollout_mse_series_falls_back_to_netcdf(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("time_days", "layers", "y", "x"), np.zeros((2, 1, 2, 2), dtype="float32")),
            "rollout_surface_thickness": (("time_days", "layers", "y", "x"), np.array([[[[2, 2], [2, 2]]], [[[3, 3], [3, 3]]]], dtype="float32")),
        },
        coords={"time_days": [1.0, 2.0], "layers": [0], "y": np.arange(2), "x": np.arange(2)},
    ).to_netcdf(rollout_path)

    assert rollout_mse_by_time_from_netcdf(rollout_path) == [4.0, 9.0]
    assert rollout_mse_series({"rollout_path": str(rollout_path)}) == [4.0, 9.0]
    assert rollout_mse_series({"eval_mse_by_time": [1, 2]}) == [1.0, 2.0]
    assert rollout_mse_series({"rollout_path": str(tmp_path / "missing.nc")}) == []
    assert rollout_mse_series({}) == []
    assert rollout_surface_thickness_rms_by_time_from_netcdf(rollout_path) == [2.0, 3.0]
    assert truth_surface_thickness_rms_by_time_from_netcdf(rollout_path) == [0.0, 0.0]
    assert surface_thickness_rms_series({"rollout_path": str(rollout_path), "eval_mse_by_time": [100.0]}) == [2.0, 3.0]
    assert surface_thickness_rms_series({"rollout_path": str(tmp_path / "missing.nc")}) == []
    assert surface_thickness_rms_series({"eval_mse_by_time": [4.0, 9.0]}) == []


def test_overview_eval_mean_mse_figure_plots_final_eval_mse_by_experiment_index(tmp_path):
    first_log_path = tmp_path / "family" / "logs" / "exp1.json"
    second_log_path = tmp_path / "family" / "logs" / "exp2.json"
    third_log_path = tmp_path / "family" / "logs" / "exp3.json"
    write_log(first_log_path, experiment_id="exp1", rollouts=[{"epoch": 5, "eval_mse_by_time": [2.0, 4.0]}])
    write_log(second_log_path, experiment_id="exp2", rollouts=[{"epoch": 5, "eval_mse_by_time": [3.0, 5.0]}])
    write_log(third_log_path, experiment_id="exp3", rollouts=[{"epoch": 5, "eval_mse_by_time": [1.0, 1.0]}])
    records = [
        TrainingRunRecord("family", "exp1", first_log_path, 1.0),
        TrainingRunRecord("family", "exp2", second_log_path, 2.0),
        TrainingRunRecord("family", "exp3", third_log_path, 3.0),
    ]

    figure = overview_eval_mean_mse_figure(records)

    assert len(figure.data) == 1
    assert figure.data[0].mode == "markers"
    assert list(figure.data[0].x) == [1, 2, 3]
    assert list(figure.data[0].y) == [3.0, 4.0, 1.0]
    assert list(figure.data[0].marker.color) == ["red", "#2563eb", "red"]


def test_overview_eval_mean_mse_figure_pins_off_scale_runs_to_axis_top(tmp_path):
    # Diverged runs (e.g. 3e27) must not be sent as y coordinates: some browsers
    # (Safari) then fail to draw every marker in the trace while hover still works.
    values = {"exp1": [1e27, 3e27], "exp2": [float("inf"), 1.0], "exp3": [100.0, 200.0]}
    records = []
    for timestamp, (experiment_id, series) in enumerate(values.items(), start=1):
        log_path = tmp_path / "family" / "logs" / f"{experiment_id}.json"
        write_log(log_path, experiment_id=experiment_id, rollouts=[{"epoch": 5, "eval_mse_by_time": series}])
        records.append(TrainingRunRecord("family", experiment_id, log_path, float(timestamp)))

    trace = overview_eval_mean_mse_figure(records).data[0]

    assert list(trace.y) == [MAX_PLOT_Y, MAX_PLOT_Y, 150.0]
    assert list(trace.marker.symbol) == ["triangle-up", "triangle-up", "circle"]
    assert "2e+27" in trace.hovertext[0] and "off scale" in trace.hovertext[0]
    assert "inf" in trace.hovertext[1]
    assert trace.marker.line.width > 0


def test_overview_eval_mean_mse_figure_skips_records_without_valid_rollouts(tmp_path):
    log_path = tmp_path / "family" / "logs" / "exp.json"
    write_log(log_path, rollouts=["bad", {"epoch": 1}])
    record = TrainingRunRecord("family", "exp", log_path, log_path.stat().st_mtime)

    trace = overview_eval_mean_mse_figure([record]).data[0]
    assert list(trace.x) == []
    assert list(trace.y) == []


def test_rollout_metrics_figure_skips_invalid_rollout_entries():
    figure = rollout_metrics_figure({"rollouts": ["bad", {"epoch": 1}, {"eval_mse_by_time": [1.0]}]})

    assert len(figure.data) == 0


def test_surface_thickness_rms_figure_requires_rollout_file_for_rms_series():
    figure = surface_thickness_rms_figure({"rollouts": [{"epoch": 5, "eval_mse_by_time": [4.0, 9.0]}]})

    assert len(figure.data) == 0
    assert figure.layout.title.text == "Surface thickness RMS by eval timestep"


def test_surface_thickness_rms_figure_adds_rollout_and_truth_rms_from_rollout_file(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("time_days", "layers", "y", "x"), np.array([[[[3, 4], [0, 0]]], [[[6, 8], [0, 0]]]], dtype="float32")),
            "rollout_surface_thickness": (("time_days", "layers", "y", "x"), np.array([[[[0, 0], [6, 8]]], [[[0, 0], [12, 16]]]], dtype="float32")),
        },
        coords={"time_days": [1.0, 2.0], "layers": [0], "y": np.arange(2), "x": np.arange(2)},
    ).to_netcdf(rollout_path)

    figure = surface_thickness_rms_figure({"rollouts": [{"epoch": 5, "rollout_path": str(rollout_path)}]})

    assert len(figure.data) == 2
    assert list(figure.data[0].y) == [5.0, 10.0]
    assert figure.data[0].name == "epoch 5 rollout RMS"
    assert figure.data[0].marker.symbol == "circle"
    assert list(figure.data[1].y) == [2.5, 5.0]
    assert figure.data[1].name == "truth RMS"
    assert figure.data[1].line.color == "white"
    assert figure.data[1].marker.color == "white"
    assert figure.data[1].line.dash == "dash"


def test_surface_thickness_rms_figure_skips_invalid_rollout_entries():
    figure = surface_thickness_rms_figure({"rollouts": ["bad", {"epoch": 1}, {"eval_mse_by_time": [4.0]}]})

    assert len(figure.data) == 0


def test_final_rollout_heatmap_figure_reads_surface_thickness_netcdf(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("time_days", "layers", "y", "x"), np.ones((2, 1, 3, 4), dtype="float32")),
            "rollout_surface_thickness": (("time_days", "layers", "y", "x"), np.zeros((2, 1, 3, 4), dtype="float32")),
        },
        coords={"time_days": [1.0, 2.0], "layers": [0], "y": np.arange(3), "x": np.arange(4)},
    ).to_netcdf(rollout_path)

    figure = final_rollout_heatmap_figure(rollout_path)

    assert len(figure.data) == 2
    assert figure.data[0].z.shape == (3, 4)


def test_load_surface_thickness_rollout_reads_full_time_series(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("time_days", "layers", "y", "x"), np.ones((3, 1, 2, 2), dtype="float32")),
            "rollout_surface_thickness": (("time_days", "layers", "y", "x"), np.zeros((3, 1, 2, 2), dtype="float32")),
        },
        coords={"time_days": [1.0, 2.0, 3.0], "layers": [0], "y": np.arange(2), "x": np.arange(2)},
    ).to_netcdf(rollout_path)

    data = load_surface_thickness_rollout(rollout_path)

    assert data.truth.shape == (3, 2, 2)
    assert data.rollout.shape == (3, 2, 2)
    assert list(data.time_days) == [1.0, 2.0, 3.0]
    assert data.zmin == 0.0
    assert data.zmax == 1.0


def test_load_surface_thickness_rollout_supports_two_layer_rollouts(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    truth = np.array(
        [
            [[[1, 2], [3, 4]], [[101, 102], [103, 104]]],
            [[[5, 6], [7, 8]], [[105, 106], [107, 108]]],
        ],
        dtype="float32",
    )
    rollout = truth + 10.0
    xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "layers", "y", "x"), truth),
            "rollout_layer_thickness": (("time_days", "layers", "y", "x"), rollout),
        },
        coords={"time_days": [1.0, 2.0], "layers": [0, 1], "y": np.arange(2), "x": np.arange(2)},
    ).to_netcdf(rollout_path)

    data = load_surface_thickness_rollout(rollout_path)

    assert data.truth.shape == (2, 2, 2)
    assert np.array_equal(data.truth[0], truth[0, 0])
    assert np.array_equal(data.rollout[1], rollout[1, 0])


def test_surface_thickness_frame_figure_plots_requested_timestep(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    truth = np.arange(24, dtype="float32").reshape(3, 1, 2, 4)
    rollout = truth + 10.0
    xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("time_days", "layers", "y", "x"), truth),
            "rollout_surface_thickness": (("time_days", "layers", "y", "x"), rollout),
        },
        coords={"time_days": [1.0, 2.5, 4.0], "layers": [0], "y": np.arange(2), "x": np.arange(4)},
    ).to_netcdf(rollout_path)

    figure = surface_thickness_frame_figure(load_surface_thickness_rollout(rollout_path), 1)

    assert len(figure.data) == 2
    assert np.array_equal(figure.data[0].z, truth[1, 0])
    assert np.array_equal(figure.data[1].z, rollout[1, 0])
    assert "2.5000" in figure.layout.title.text
    assert figure.layout.coloraxis.cmin == 0.0
    assert figure.layout.coloraxis.cmax == 33.0


def test_surface_thickness_animation_figure_builds_plotly_frames(tmp_path):
    rollout_path = tmp_path / "rollout.nc"
    truth = np.arange(24, dtype="float32").reshape(3, 1, 2, 4)
    rollout = truth + 10.0
    xr.Dataset(
        data_vars={
            "truth_surface_thickness": (("time_days", "layers", "y", "x"), truth),
            "rollout_surface_thickness": (("time_days", "layers", "y", "x"), rollout),
        },
        coords={"time_days": [1.0, 2.5, 4.0], "layers": [0], "y": np.arange(2), "x": np.arange(4)},
    ).to_netcdf(rollout_path)

    figure = surface_thickness_animation_figure(load_surface_thickness_rollout(rollout_path), frame_duration_ms=200)

    assert len(figure.data) == 2
    assert len(figure.frames) == 3
    assert figure.frames[1].name == "timestep-1"
    assert list(figure.frames[1].traces) == [0, 1]
    assert np.array_equal(figure.frames[1].data[0].z, truth[1, 0])
    assert figure.layout.coloraxis.cmin == 0.0
    assert figure.layout.coloraxis.cmax == 33.0
    assert figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] is True
    assert figure.layout.updatemenus[0].buttons[0].label == "Play"
