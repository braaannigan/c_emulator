"""Submit dgsw2l experiments to the RunPod endpoint and record their results locally."""

from __future__ import annotations

import json
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib import request

API_BASE = "https://api.runpod.ai/v2"
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

Transport = Callable[[str, str, dict[str, str], bytes | None], dict[str, Any]]


def read_api_key(env: dict[str, str], config_path: Path) -> str:
    """API key from ``RUNPOD_API_KEY`` or the runpodctl config file."""

    if env.get("RUNPOD_API_KEY"):
        return env["RUNPOD_API_KEY"]
    if config_path.exists():
        key = tomllib.loads(config_path.read_text()).get("apikey")
        if key:
            return str(key)
    raise RuntimeError("no RunPod API key: set RUNPOD_API_KEY or run `runpodctl doctor`")


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> dict[str, Any]:  # pragma: no cover - network
    req = request.Request(url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


class EndpointClient:
    def __init__(self, endpoint_id: str, api_key: str, transport: Transport = urllib_transport, base: str = API_BASE):
        self.url = f"{base}/{endpoint_id}"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.transport = transport

    def submit(self, job_input: dict[str, Any]) -> str:
        reply = self.transport("POST", f"{self.url}/run", self.headers, json.dumps({"input": job_input}).encode())
        return str(reply["id"])

    def status(self, job_id: str) -> dict[str, Any]:
        return self.transport("GET", f"{self.url}/status/{job_id}", self.headers, None)

    def wait(self, job_id: str, poll_seconds: float = 30.0, sleep: Callable[[float], None] = time.sleep, on_status: Callable[[str], None] = lambda s: None) -> dict[str, Any]:
        while True:
            reply = self.status(job_id)
            on_status(str(reply.get("status")))
            if reply.get("status") in TERMINAL_STATUSES:
                return reply
            sleep(poll_seconds)


def record_outputs(output: dict[str, Any], repo_root: Path, results_csv: Path) -> list[str]:
    """Write the returned experiment log and append its results rows locally; return written paths."""

    written = []
    if output.get("experiment_log") is not None:
        log_path = repo_root / output["experiment_log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(output["experiment_log"], indent=2) + "\n")
        written.append(str(log_path))
    rows = output.get("results_rows") or []
    if rows:
        if not results_csv.exists():
            raise FileNotFoundError(f"local results CSV {results_csv} is missing; cannot append remote rows")
        with results_csv.open("a") as handle:
            handle.write("".join(row + "\n" for row in rows))
        written.append(str(results_csv))
    return written
