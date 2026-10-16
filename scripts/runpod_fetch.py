"""Fetch a RunPod experiment's outputs from the network volume.

Usage:
    uv run python scripts/runpod_fetch.py <experiment-id> [<experiment-id> ...] [--checkpoint]

Starts a short-lived CPU pod in the volume's datacenter, copies each run's
experiment log and diagnostics report (and optionally its checkpoint) from
``outputs/<id>/`` on the volume into ``data/double_gyre_shifting_wind_2layer``,
then deletes the pod. Use it when a job's result was lost by the API.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "data" / "double_gyre_shifting_wind_2layer"
KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"


def runpodctl(*args: str) -> dict | list:
    return json.loads(subprocess.run(["runpodctl", *args], check=True, capture_output=True, text=True).stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RunPod experiment outputs from the network volume")
    parser.add_argument("experiment_ids", nargs="+")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    endpoint = json.loads((ROOT / "deploy" / "runpod" / "endpoint.json").read_text())
    pod = runpodctl("pod", "create", "--compute-type", "CPU", "--template-id", "runpod-ubuntu-2204", "--name", "dgsw2l-fetch",
                    "--data-center-ids", endpoint["data_center"], "--network-volume-id", endpoint["network_volume_id"], "--wait", "--wait-timeout", "8m")
    try:
        ssh = pod["ssh"]
        opts = ["-q", "-i", str(KEY), "-P", str(ssh["port"]), "-o", "StrictHostKeyChecking=accept-new"]
        host = f"root@{ssh['ip']}"
        for experiment_id in args.experiment_ids:
            remote = f"/workspace/outputs/{experiment_id}"
            targets = [(f"{remote}/{experiment_id}.json", FAMILY / "logs"), (f"{remote}/diagnostics/report.json", FAMILY / "diagnostics" / experiment_id)]
            if args.checkpoint:
                targets.append((f"{remote}/{experiment_id}.pt", FAMILY / "checkpoints"))
            for source, destination in targets:
                destination.mkdir(parents=True, exist_ok=True)
                done = subprocess.run(["scp", *opts, f"{host}:{source}", str(destination) + "/"]).returncode == 0
                print(f"{'fetched' if done else 'missing'} {source}")
    finally:
        subprocess.run(["runpodctl", "pod", "delete", pod["id"]], capture_output=True)
        print(f"deleted pod {pod['id']}")


if __name__ == "__main__":
    main()
