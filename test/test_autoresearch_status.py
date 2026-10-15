import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "autoresearch_status.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_status", MODULE_PATH)
autoresearch_status = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(autoresearch_status)

config_path_for_experiment = autoresearch_status.config_path_for_experiment
experiment_kind_from_config = autoresearch_status.experiment_kind_from_config
final_rollout_mean = autoresearch_status.final_rollout_mean
paths_for_family = autoresearch_status.paths_for_family
recent_experiment_kind_mix = autoresearch_status.recent_experiment_kind_mix
results_rows = autoresearch_status.results_rows
scheduled_rollout_summary = autoresearch_status.scheduled_rollout_summary


def test_final_rollout_mean_and_schedule(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {"epoch": 5, "eval_mse_by_time": [1.0, 3.0], "eval_mse_last": 3.0},
                    {"epoch": 10, "eval_mse_by_time": [2.0, 6.0], "eval_mse_last": 6.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert final_rollout_mean(log_path) == 4.0
    assert scheduled_rollout_summary(log_path) == [(5, 2.0, 3.0), (10, 4.0, 6.0)]


def test_final_rollout_mean_handles_missing_series(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps({"rollouts": [{"epoch": 5}, "bad"]}), encoding="utf-8")

    assert final_rollout_mean(log_path) is None
    assert scheduled_rollout_summary(log_path) == []


def test_rollout_helpers_handle_non_list_rollouts_and_missing_last(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps({"rollouts": {"bad": "shape"}}), encoding="utf-8")
    assert final_rollout_mean(log_path) is None

    log_path.write_text(json.dumps({"rollouts": [{"epoch": 1, "eval_mse_by_time": [2.0]}]}), encoding="utf-8")
    assert scheduled_rollout_summary(log_path) == [(1, 2.0, None)]


def test_results_rows_reads_configured_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "eval_mse_mean"])
        writer.writeheader()
        writer.writerow({"experiment_id": "exp", "eval_mse_mean": "1.0"})
    monkeypatch.setattr(autoresearch_status, "RESULTS_CSV", csv_path)

    assert results_rows(csv_path) == [{"experiment_id": "exp", "eval_mse_mean": "1.0"}]


def test_results_rows_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(autoresearch_status, "RESULTS_CSV", tmp_path / "missing.csv")

    assert results_rows(tmp_path / "missing.csv") == []


def test_paths_for_family_returns_data_results_and_logs_paths():
    results_csv, log_dir = paths_for_family("family")

    assert results_csv == Path("data/family/results.csv")
    assert log_dir == Path("data/family/logs")


def test_experiment_kind_from_config_reads_kind_in_experiment_block(tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text(
        "experiment:\n"
        "  id: exp\n"
        "  kind: medium_step\n"
        "  rationale: >\n"
        "    Testing a larger objective change.\n"
        "paths:\n"
        "  training_data: data.nc\n",
        encoding="utf-8",
    )

    assert experiment_kind_from_config(config) == "medium_step"


def test_experiment_kind_from_config_handles_missing_and_unknown_kind(tmp_path):
    missing = tmp_path / "missing.yaml"
    missing.write_text("experiment:\n  id: exp\npaths:\n  training_data: data.nc\n", encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("experiment:\n  kind: tiny_tweak\n", encoding="utf-8")

    assert experiment_kind_from_config(missing) == "unknown"
    assert experiment_kind_from_config(unknown) == "unknown"


def test_config_path_for_experiment_finds_family_config(tmp_path):
    config = tmp_path / "family" / "exp.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("experiment:\n  kind: tuning\n", encoding="utf-8")

    assert config_path_for_experiment("exp", tmp_path) == config
    assert config_path_for_experiment("missing", tmp_path) is None


def test_recent_experiment_kind_mix_reports_counts_streak_and_recommendation(tmp_path):
    log_dir = tmp_path / "logs"
    config_root = tmp_path / "configs"
    log_dir.mkdir()
    for index, (experiment_id, kind) in enumerate([("old", "bigger_jump"), ("mid", "tuning"), ("new", "tuning")], start=1):
        log = log_dir / f"{experiment_id}.json"
        log.write_text(json.dumps({"metadata": {"experiment_id": experiment_id}}), encoding="utf-8")
        config = config_root / "family" / f"{experiment_id}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"experiment:\n  kind: {kind}\n", encoding="utf-8")
        stamp = float(index)
        log.touch()
        import os

        os.utime(log, (stamp, stamp))

    mix = recent_experiment_kind_mix(log_dir, config_root, limit=3)

    assert mix["kinds"] == ["tuning", "tuning", "bigger_jump"]
    assert mix["counts"] == {"tuning": 2, "medium_step": 0, "bigger_jump": 1, "unknown": 0}
    assert mix["streak_kind"] == "tuning"
    assert mix["streak_length"] == 2
    assert mix["recommendation"] == "next experiment should be medium_step or bigger_jump"


def test_recent_experiment_kind_mix_forces_bigger_jump_when_recent_window_has_none(tmp_path):
    log_dir = tmp_path / "logs"
    config_root = tmp_path / "configs"
    log_dir.mkdir()
    for index, experiment_id in enumerate(["a", "b", "c", "d"], start=1):
        log = log_dir / f"{experiment_id}.json"
        log.write_text(json.dumps({"metadata": {"experiment_id": experiment_id}}), encoding="utf-8")
        config = config_root / "family" / f"{experiment_id}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("experiment:\n  kind: medium_step\n", encoding="utf-8")
        import os

        os.utime(log, (float(index), float(index)))

    mix = recent_experiment_kind_mix(log_dir, config_root, limit=4)

    assert mix["counts"]["bigger_jump"] == 0
    assert mix["recommendation"] == "next experiment should be bigger_jump unless recent evidence strongly justifies otherwise"


def test_recent_experiment_kind_mix_handles_empty_and_missing_config(tmp_path):
    log_dir = tmp_path / "logs"
    config_root = tmp_path / "configs"
    log_dir.mkdir()
    assert recent_experiment_kind_mix(log_dir, config_root)["streak_kind"] is None

    log = log_dir / "exp.json"
    log.write_text(json.dumps({"metadata": {"experiment_id": "exp"}}), encoding="utf-8")
    mix = recent_experiment_kind_mix(log_dir, config_root)

    assert mix["kinds"] == ["unknown"]
    assert mix["counts"]["unknown"] == 1
