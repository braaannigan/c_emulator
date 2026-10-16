"""Run a queue of dgsw2l jobs on a rented GPU pod, several at a time.

Usage (on the pod):
    python -m src.remote.pod_worker <jobs.json> --volume <dir> --work <dir> [--concurrency 3]

``jobs.json`` is a list of job inputs as accepted by ``src.remote.job.run_job``
(``ref``, ``config``, optional ``diagnostics``/``long_years``). Each job's
result is written to ``<volume>/outputs/<experiment-id or index>/result.json``
next to the outputs ``run_job`` already copies there, and ``<volume>/DONE`` is
written when the whole queue has finished, so a local orchestrator can poll
and copy results back. A small model leaves most of a GPU idle, so running a
few jobs concurrently costs little extra time per job.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.remote.job import json_safe, run_job

JobRunner = Callable[[dict[str, Any], Path, Path], dict[str, Any]]


def run_queue(jobs: list[dict[str, Any]], volume_dir: Path, work_dir: Path, concurrency: int = 3, runner: JobRunner = run_job) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    def one(index_job: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        index, job = index_job
        try:
            result = runner(job, work_dir / f"job{index}", volume_dir)
        except Exception as error:  # keep the rest of the queue running
            result = {"error": repr(error), "config": job.get("config")}
        name = result.get("experiment_id") or f"job{index}"
        out = volume_dir / "outputs" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, enumerate(jobs)))
    (volume_dir / "DONE").write_text(json.dumps([r.get("experiment_id") or r.get("error") for r in results]) + "\n")
    return results


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description="Run a queue of dgsw2l jobs on a GPU pod")
    parser.add_argument("jobs")
    parser.add_argument("--volume", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(argv)
    run_queue(json.loads(Path(args.jobs).read_text()), Path(args.volume), Path(args.work), args.concurrency)


if __name__ == "__main__":  # pragma: no cover
    main()
