"""Run one dgsw2l experiment on a RunPod serverless worker.

The worker clones the repository at the requested git ref, links the training
data from the network volume into the clone, runs the pipeline CLI, copies the
outputs back to the volume and returns the experiment log and results rows so
the local machine can record them.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

REPO_URL = "https://github.com/braaannigan/c_emulator.git"
FAMILY_DIR = Path("data/double_gyre_shifting_wind_2layer")

Runner = Callable[..., subprocess.CompletedProcess]


def validate_job(job_input: dict[str, Any]) -> tuple[str, str, bool]:
    """Return ``(ref, config, smoke_test)`` from a job's input, rejecting bad values."""

    ref = str(job_input.get("ref", "")).strip()
    config = str(job_input.get("config", "")).strip()
    if not ref or ref.startswith("-"):
        raise ValueError("job input needs a git ref (branch or commit)")
    if not config.startswith("configs/") or ".." in Path(config).parts:
        raise ValueError("config must be a path under configs/")
    return ref, config, bool(job_input.get("smoke_test", False))


def clone_repo(ref: str, dest: Path, run: Runner = subprocess.run, repo_url: str = REPO_URL) -> str:
    """Clone ``repo_url`` at ``ref`` into ``dest`` and return the checked-out commit."""

    run(["git", "clone", "--quiet", "--filter=blob:none", repo_url, str(dest)], check=True)
    run(["git", "-C", str(dest), "checkout", "--quiet", ref], check=True)
    head = run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return head.stdout.strip()


def link_training_data(repo_dir: Path, volume_dir: Path) -> Path:
    """Point the clone's training-data directory at the copy on the network volume."""

    source = volume_dir / FAMILY_DIR / "train"
    if not source.is_dir():
        raise FileNotFoundError(f"training data not found on volume at {source}")
    target = repo_dir / FAMILY_DIR / "train"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    target.symlink_to(source, target_is_directory=True)
    return target


def run_pipeline(repo_dir: Path, config: str, smoke_test: bool, run: Runner = subprocess.run) -> subprocess.CompletedProcess:
    command = ["python", "-u", "-m", "src.pipelines.dgsw2l_baseline", "--config", config]
    if smoke_test:
        command.append("--smoke-test")
    return run(command, cwd=repo_dir, capture_output=True, text=True)


def collect_outputs(repo_dir: Path, config: str, volume_dir: Path) -> dict[str, Any]:
    """Copy the run's outputs to ``<volume>/outputs/<id>/`` and return the log and results rows."""

    raw = yaml.safe_load((repo_dir / config).read_text())
    experiment_id = raw["experiment"]["id"]
    out_dir = volume_dir / "outputs" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = repo_dir / raw["paths"]["experiment_log"]
    results_path = repo_dir / raw["paths"]["results_csv"]
    checkpoint = results_path.parent / "checkpoints" / f"{experiment_id}.pt"
    rollouts = sorted((repo_dir / raw["paths"]["rollout_dir"]).glob(f"{experiment_id}*.nc"))
    copied = []
    for path in [log_path, results_path, checkpoint, *rollouts]:
        if path.exists():
            shutil.copy2(path, out_dir / path.name)
            copied.append(str(out_dir / path.name))
    rows = results_path.read_text().splitlines()[1:] if results_path.exists() else []
    return {
        "experiment_id": experiment_id,
        "experiment_log_path": raw["paths"]["experiment_log"],
        "experiment_log": json.loads(log_path.read_text()) if log_path.exists() else None,
        "results_rows": rows,
        "volume_outputs": copied,
    }


def run_diagnostics(repo_dir: Path, config: str, long_years: float, volume_dir: Path, run: Runner = subprocess.run) -> dict[str, Any]:
    """Run the fidelity diagnostics on the freshly trained checkpoint; keep figures on the volume."""

    command = ["python", "-u", "scripts/fidelity_diagnostics.py", "--config", config, "--long-years", str(long_years)]
    proc = run(command, cwd=repo_dir, capture_output=True, text=True)
    raw = yaml.safe_load((repo_dir / config).read_text())
    experiment_id = raw["experiment"]["id"]
    out_dir = repo_dir / Path(raw["paths"]["results_csv"]).parent / "diagnostics" / experiment_id
    result: dict[str, Any] = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}
    report = out_dir / "report.json"
    if proc.returncode == 0 and report.exists():
        result["report"] = json.loads(report.read_text())
        shutil.copytree(out_dir, volume_dir / "outputs" / experiment_id / "diagnostics", dirs_exist_ok=True)
    return result


def json_safe(value: Any) -> Any:
    """Replace non-finite floats (NaN, +/-inf) with None, recursively.

    RunPod drops a job's whole output if it is not strict JSON, and
    diagnostics can legitimately contain inf (e.g. an e-folding time that
    never crosses 1/e).
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def run_job(job_input: dict[str, Any], work_dir: Path, volume_dir: Path, run: Runner = subprocess.run) -> dict[str, Any]:
    ref, config, smoke_test = validate_job(job_input)
    repo_dir = work_dir / "repo"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    commit = clone_repo(ref, repo_dir, run)
    link_training_data(repo_dir, volume_dir)
    proc = run_pipeline(repo_dir, config, smoke_test, run)
    result: dict[str, Any] = {
        "ref": ref,
        "commit": commit,
        "config": config,
        "smoke_test": smoke_test,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode == 0:
        result.update(collect_outputs(repo_dir, config, volume_dir))
        if job_input.get("diagnostics") and not smoke_test:
            result["diagnostics"] = run_diagnostics(repo_dir, config, float(job_input.get("long_years", 25)), volume_dir, run)
    return json_safe(result)
