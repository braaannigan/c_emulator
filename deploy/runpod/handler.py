"""RunPod serverless entry point: one job = one dgsw2l experiment (see src/remote/job.py)."""

import os
from pathlib import Path

import runpod

from src.remote.job import run_job

VOLUME_DIR = Path(os.environ.get("VOLUME_DIR", "/runpod-volume"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/dgsw2l-job"))


def handler(job):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return run_job(job["input"], WORK_DIR, VOLUME_DIR)


runpod.serverless.start({"handler": handler})
