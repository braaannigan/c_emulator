import json
import threading
import time

import pytest

from src.remote.pod_worker import run_queue


def test_run_queue_runs_jobs_concurrently_and_records_results(tmp_path):
    active, peak = [0], [0]
    lock = threading.Lock()

    def runner(job, work_dir, volume_dir):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        if job["config"] == "boom":
            raise RuntimeError("bad job")
        return {"experiment_id": job["config"], "score": float("inf"), "work": str(work_dir)}

    jobs = [{"config": "a"}, {"config": "b"}, {"config": "boom"}]
    results = run_queue(jobs, tmp_path / "vol", tmp_path / "work", concurrency=3, runner=runner)
    assert peak[0] >= 2
    assert json.loads((tmp_path / "vol" / "outputs" / "a" / "result.json").read_text())["score"] is None
    failed = json.loads((tmp_path / "vol" / "outputs" / "job2" / "result.json").read_text())
    assert "bad job" in failed["error"] and failed["config"] == "boom"
    assert json.loads((tmp_path / "vol" / "DONE").read_text())[:2] == ["a", "b"]
    assert results[1]["work"].endswith("job1")
    with pytest.raises(ValueError):
        run_queue(jobs, tmp_path, tmp_path, concurrency=0, runner=runner)


def test_run_queue_retries_out_of_memory_jobs_serially(tmp_path):
    calls = []

    def runner(job, work_dir, volume_dir):
        calls.append(job["config"])
        if job["config"] == "big" and calls.count("big") == 1:
            return {"returncode": 1, "stderr_tail": "torch.OutOfMemoryError: CUDA out of memory"}
        return {"experiment_id": job["config"], "returncode": 0}

    results = run_queue([{"config": "a"}, {"config": "big"}], tmp_path / "vol", tmp_path / "work", concurrency=2, runner=runner)
    assert calls.count("big") == 2
    assert results[1] == {"experiment_id": "big", "returncode": 0}
    assert json.loads((tmp_path / "vol" / "outputs" / "big" / "result.json").read_text())["returncode"] == 0
