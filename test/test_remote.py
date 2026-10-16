import json
import subprocess
from pathlib import Path

import pytest
import yaml

from src.remote.client import EndpointClient, read_api_key, record_outputs
from src.remote.job import FAMILY_DIR, clone_repo, collect_outputs, link_training_data, run_job, run_pipeline, validate_job

CONFIG = "configs/double_gyre_shifting_wind_2layer/exp.yaml"


def write_repo(repo: Path, experiment_id: str = "exp") -> None:
    cfg = repo / CONFIG
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({
        "experiment": {"id": experiment_id},
        "paths": {
            "experiment_log": f"{FAMILY_DIR}/logs/{experiment_id}.json",
            "results_csv": f"{FAMILY_DIR}/results.csv",
            "rollout_dir": f"{FAMILY_DIR}/rollout/",
        },
    }))


def write_outputs(repo: Path, experiment_id: str = "exp") -> None:
    family = repo / FAMILY_DIR
    for sub in ["logs", "checkpoints", "rollout"]:
        (family / sub).mkdir(parents=True, exist_ok=True)
    (family / "logs" / f"{experiment_id}.json").write_text(json.dumps({"metadata": {"x": 1}}))
    (family / "results.csv").write_text("header\nrow1\n")
    (family / "checkpoints" / f"{experiment_id}.pt").write_bytes(b"ckpt")
    (family / "rollout" / f"{experiment_id}-final.nc").write_bytes(b"nc")


def make_volume(tmp_path: Path) -> Path:
    volume = tmp_path / "volume"
    (volume / FAMILY_DIR / "train").mkdir(parents=True)
    (volume / FAMILY_DIR / "train" / "data.nc").write_bytes(b"data")
    return volume


@pytest.mark.parametrize("bad", [{}, {"ref": "-x", "config": CONFIG}, {"ref": "main", "config": "/etc/passwd"}, {"ref": "main", "config": "configs/../secret.yaml"}])
def test_validate_job_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_job(bad)


def test_validate_job_defaults_smoke_test_off():
    assert validate_job({"ref": " main ", "config": CONFIG}) == ("main", CONFIG, False)
    assert validate_job({"ref": "main", "config": CONFIG, "smoke_test": True})[2] is True


def test_clone_repo_checks_out_ref_from_local_repository(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "--quiet", "-b", "main", str(src)], check=True)
    (src / "a.txt").write_text("a")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "--quiet", "-m", "a"], check=True)
    subprocess.run([*git, "switch", "--quiet", "-c", "exp"], check=True)
    (src / "a.txt").write_text("b")
    subprocess.run([*git, "commit", "--quiet", "-am", "b"], check=True)
    expected = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

    commit = clone_repo("exp", tmp_path / "clone", repo_url=str(src))

    assert commit == expected
    assert (tmp_path / "clone" / "a.txt").read_text() == "b"


def test_link_training_data_replaces_existing_entries(tmp_path):
    volume = make_volume(tmp_path)
    repo = tmp_path / "repo"
    target = repo / FAMILY_DIR / "train"
    target.mkdir(parents=True)
    assert link_training_data(repo, volume).resolve() == (volume / FAMILY_DIR / "train").resolve()
    assert (target / "data.nc").read_bytes() == b"data"
    link_training_data(repo, volume)  # existing symlink
    target.unlink()
    target.write_text("stale file")
    link_training_data(repo, volume)
    assert target.is_symlink()
    with pytest.raises(FileNotFoundError):
        link_training_data(repo, tmp_path / "empty-volume")


def test_run_pipeline_builds_cli_command(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    run_pipeline(tmp_path, CONFIG, True, fake_run)
    run_pipeline(tmp_path, CONFIG, False, fake_run)
    assert calls[0][0][-1] == "--smoke-test" and calls[1][0][-1] == CONFIG
    assert calls[0][1]["cwd"] == tmp_path


def test_collect_outputs_copies_to_volume_and_returns_rows(tmp_path):
    repo, volume = tmp_path / "repo", tmp_path / "volume"
    write_repo(repo)
    write_outputs(repo)
    out = collect_outputs(repo, CONFIG, volume)
    assert out["experiment_id"] == "exp"
    assert out["experiment_log"] == {"metadata": {"x": 1}}
    assert out["results_rows"] == ["row1"]
    assert sorted(Path(p).name for p in out["volume_outputs"]) == ["exp-final.nc", "exp.json", "exp.pt", "results.csv"]
    assert (volume / "outputs" / "exp" / "exp.pt").read_bytes() == b"ckpt"


def test_collect_outputs_tolerates_missing_files(tmp_path):
    repo = tmp_path / "repo"
    write_repo(repo)
    out = collect_outputs(repo, CONFIG, tmp_path / "volume")
    assert out["experiment_log"] is None and out["results_rows"] == [] and out["volume_outputs"] == []


def fake_worker(tmp_path, returncode=0, diagnostics_returncode=0):
    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True)
            write_repo(dest)
        elif "rev-parse" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "abc123\n", "")
        elif cmd[0] == "python" and "scripts/fidelity_diagnostics.py" in cmd:
            out = kwargs["cwd"] / FAMILY_DIR / "diagnostics" / "exp"
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text(json.dumps({"held_out": {"eval_mse_mean": 1.0}}))
            (out / "maps.png").write_bytes(b"png")
            return subprocess.CompletedProcess(cmd, diagnostics_returncode, "diag ok", "")
        elif cmd[0] == "python":
            write_outputs(kwargs["cwd"])
            return subprocess.CompletedProcess(cmd, returncode, "x" * 5000 + "done", "warn")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run


def test_run_job_end_to_end_with_fake_commands(tmp_path):
    volume = make_volume(tmp_path)
    (tmp_path / "work" / "repo").mkdir(parents=True)  # stale clone from a previous job
    out = run_job({"ref": "exp", "config": CONFIG}, tmp_path / "work", volume, fake_worker(tmp_path))
    assert out["commit"] == "abc123" and out["returncode"] == 0
    assert out["stdout_tail"].endswith("done") and len(out["stdout_tail"]) == 4000
    assert out["results_rows"] == ["row1"]


def test_run_job_failure_skips_output_collection(tmp_path):
    out = run_job({"ref": "exp", "config": CONFIG}, tmp_path / "work", make_volume(tmp_path), fake_worker(tmp_path, returncode=1))
    assert out["returncode"] == 1 and "results_rows" not in out


def test_read_api_key_prefers_env_then_config(tmp_path):
    cfg = tmp_path / "config.toml"
    with pytest.raises(RuntimeError):
        read_api_key({}, cfg)
    cfg.write_text('apikey = ""\n')
    with pytest.raises(RuntimeError):
        read_api_key({}, cfg)
    cfg.write_text('apikey = "from-file"\n')
    assert read_api_key({}, cfg) == "from-file"
    assert read_api_key({"RUNPOD_API_KEY": "from-env"}, cfg) == "from-env"


def test_endpoint_client_submits_and_polls_until_terminal():
    calls = []
    statuses = iter([{"status": "IN_QUEUE"}, {"status": "IN_PROGRESS"}, {"status": "COMPLETED", "output": {"ok": 1}}])

    def transport(method, url, headers, body):
        calls.append((method, url, headers, body))
        return {"id": "job1"} if method == "POST" else next(statuses)

    client = EndpointClient("ep", "key", transport, base="https://api")
    assert client.submit({"ref": "main"}) == "job1"
    assert calls[0][1] == "https://api/ep/run" and json.loads(calls[0][3]) == {"input": {"ref": "main"}}
    assert calls[0][2]["Authorization"] == "Bearer key"
    seen, sleeps = [], []
    reply = client.wait("job1", 5, sleep=sleeps.append, on_status=seen.append)
    assert reply["output"] == {"ok": 1}
    assert seen == ["IN_QUEUE", "IN_PROGRESS", "COMPLETED"] and sleeps == [5, 5]
    assert calls[-1][1] == "https://api/ep/status/job1"


def test_record_outputs_writes_log_and_appends_rows(tmp_path):
    results = tmp_path / "results.csv"
    output = {"experiment_log_path": "logs/exp.json", "experiment_log": {"a": 1}, "results_rows": ["r1", "r2"]}
    with pytest.raises(FileNotFoundError):
        record_outputs(output, tmp_path, results)
    results.write_text("header\n")
    written = record_outputs(output, tmp_path, results)
    assert json.loads((tmp_path / "logs" / "exp.json").read_text()) == {"a": 1}
    assert results.read_text() == "header\nr1\nr2\n"
    assert written == [str(tmp_path / "logs" / "exp.json"), str(results)]
    assert record_outputs({"experiment_log": None}, tmp_path, results) == []


def test_run_job_with_diagnostics_returns_report_and_keeps_figures(tmp_path):
    volume = make_volume(tmp_path)
    out = run_job({"ref": "exp", "config": CONFIG, "diagnostics": True, "long_years": 2}, tmp_path / "work", volume, fake_worker(tmp_path))
    assert out["diagnostics"]["report"] == {"held_out": {"eval_mse_mean": 1.0}}
    assert (volume / "outputs" / "exp" / "diagnostics" / "maps.png").read_bytes() == b"png"
    failed = run_job({"ref": "exp", "config": CONFIG, "diagnostics": True}, tmp_path / "work2", volume, fake_worker(tmp_path, diagnostics_returncode=1))
    assert failed["diagnostics"]["returncode"] == 1 and "report" not in failed["diagnostics"]
    smoke = run_job({"ref": "exp", "config": CONFIG, "diagnostics": True, "smoke_test": True}, tmp_path / "work3", volume, fake_worker(tmp_path))
    assert "diagnostics" not in smoke


def test_record_outputs_writes_diagnostics_report(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text("header\n")
    output = {
        "experiment_id": "exp",
        "experiment_log_path": "data/fam/logs/exp.json",
        "experiment_log": {"a": 1},
        "results_rows": [],
        "diagnostics": {"report": {"k": 2}},
    }
    written = record_outputs(output, tmp_path, results)
    report = tmp_path / "data" / "fam" / "diagnostics" / "exp" / "report.json"
    assert json.loads(report.read_text()) == {"k": 2} and str(report) in written
