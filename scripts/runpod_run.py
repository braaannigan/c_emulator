"""Run a dgsw2l experiment on the RunPod endpoint and record its outputs locally.

Usage:
    uv run python scripts/runpod_run.py --config <config> [--ref <branch>] [--smoke-test]

The ref (default: the current branch) must be pushed to GitHub, since the worker
clones it. Outputs (log, checkpoint, rollouts) are also kept on the network
volume under outputs/<experiment-id>/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.remote.client import EndpointClient, read_api_key, record_outputs  # noqa: E402

ENDPOINT_FILE = ROOT / "deploy" / "runpod" / "endpoint.json"
RESULTS_CSV = ROOT / "data" / "double_gyre_shifting_wind_2layer" / "results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--diagnostics", action="store_true", help="Also run the fidelity diagnostics remotely and save the report locally")
    parser.add_argument("--long-years", type=float, default=25.0)
    args = parser.parse_args()
    ref = args.ref or subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
    endpoint_id = json.loads(ENDPOINT_FILE.read_text())["endpoint_id"]
    client = EndpointClient(endpoint_id, read_api_key(dict(os.environ), Path.home() / ".runpod" / "config.toml"))
    job_id = client.submit({"ref": ref, "config": args.config, "smoke_test": args.smoke_test, "diagnostics": args.diagnostics, "long_years": args.long_years})
    print(f"submitted job {job_id} (ref {ref})", flush=True)
    last = [""]

    def show(status: str) -> None:
        if status != last[0]:
            print(f"status: {status}", flush=True)
            last[0] = status

    reply = client.wait(job_id, args.poll_seconds, on_status=show)
    output = reply.get("output") or {}
    if reply.get("status") != "COMPLETED" or output.get("returncode") != 0:
        print(json.dumps({k: v for k, v in reply.items() if k != "output"}, indent=2))
        print(output.get("stdout_tail", ""), output.get("stderr_tail", ""), sep="\n")
        sys.exit(1)
    if args.smoke_test:
        print("smoke test passed; not recording outputs locally")
    else:
        for path in record_outputs(output, ROOT, RESULTS_CSV):
            print(f"wrote {path}")
    print(output.get("stdout_tail", "").strip().splitlines()[-4:])
    print(f"execution {reply.get('executionTime', 0) / 1000:.0f} s, queue delay {reply.get('delayTime', 0) / 1000:.0f} s, commit {output.get('commit')}")


if __name__ == "__main__":
    main()
