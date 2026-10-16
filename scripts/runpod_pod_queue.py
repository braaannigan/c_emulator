"""Run a queue of experiments on an hourly-rented RunPod GPU pod (cheapest per run).

Usage:
    uv run python scripts/runpod_pod_queue.py --job <ref>:<config> [--job <ref>:<config> ...]
        [--concurrency 3] [--diagnostics] [--checkpoints] [--gpu COMMUNITY:"NVIDIA GeForce RTX 3090" ...] [--keep]

Rents one pod (the first available of a preference list, cheapest community cards first, billed per hour), uploads the
training data once, runs the jobs ``--concurrency`` at a time on the single
GPU (the model leaves most of a GPU idle), copies each run's log, results row,
diagnostics report (and optionally checkpoint) back into
``data/double_gyre_shifting_wind_2layer``, and always deletes the pod at the
end unless ``--keep``. Compared with the serverless endpoint there is no
per-second premium and no per-job cold start.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.remote.client import record_outputs  # noqa: E402

FAMILY = ROOT / "data" / "double_gyre_shifting_wind_2layer"
TRAIN_NAME = "20260408T071500-period40d-duration5000d-2layer.nc"
REMOTE_VOLUME = "/root/volume"
KEY = Path.home() / ".runpod" / "ssh" / "runpodctl-ssh-key"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
SETUP = """set -e
python -m venv --system-site-packages /root/venv
/root/venv/bin/pip install --quiet --no-cache-dir runpod xarray netcdf4 pyyaml scipy matplotlib
rm -rf /root/c_emulator && git clone --quiet --depth 1 https://github.com/braaannigan/c_emulator.git /root/c_emulator
mkdir -p {volume}/data/double_gyre_shifting_wind_2layer/train /root/chunks
"""


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Pod:
    def __init__(self, pod: dict):
        self.id = pod["id"]
        ssh = pod["ssh"]
        self.host = f"root@{ssh['ip']}"
        self.port = str(ssh["port"])
        self.opts = ["-i", str(KEY), "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]

    def ssh(self, command: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *self.opts, "-p", self.port, self.host, command], check=check, capture_output=capture, text=True)

    def scp_to(self, local: Path, remote: str, attempts: int = 3) -> None:
        for attempt in range(attempts):
            if subprocess.run(["scp", "-q", *self.opts, "-P", self.port, str(local), f"{self.host}:{remote}"]).returncode == 0:
                return
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"scp of {local.name} failed {attempts} times")


DEFAULT_GPUS = [
    "COMMUNITY:NVIDIA GeForce RTX 4090",
    "COMMUNITY:NVIDIA GeForce RTX 3090",
    "COMMUNITY:NVIDIA RTX A5000",
    "SECURE:NVIDIA RTX A5000",
    "SECURE:NVIDIA GeForce RTX 3090",
    "SECURE:NVIDIA GeForce RTX 4090",
]


def create_pod(choices: list[str]) -> Pod:
    """Create a pod with the first available ``CLOUD:gpu-id`` choice, in order of preference."""

    for choice in choices:
        cloud, gpu = choice.split(":", 1)
        result = subprocess.run(
            ["runpodctl", "pod", "create", "--image", IMAGE, "--gpu-id", gpu, "--cloud-type", cloud, "--name", "dgsw2l-queue",
             "--container-disk-in-gb", "40", "--ports", "22/tcp", "--wait", "--wait-timeout", "6m"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            log(f"got {cloud} {gpu}")
            return Pod(json.loads(result.stdout))
        message = (result.stdout.strip() or result.stderr.strip())[:200]
        log(f"unavailable: {cloud} {gpu} ({message})")
        # A pod that was created but never became reachable over SSH still bills: delete it.
        for pod in json.loads(subprocess.run(["runpodctl", "pod", "list"], capture_output=True, text=True).stdout or "[]"):
            if pod.get("name") == "dgsw2l-queue":
                subprocess.run(["runpodctl", "pod", "delete", pod["id"]], capture_output=True)
                log(f"deleted unreachable pod {pod['id']}")
    raise RuntimeError("no GPU available from the preference list")


def upload_data(pod: Pod, pieces: int = 16, parallel: int = 8) -> None:
    """Parallel chunked upload: a single SSH stream is far slower than the link."""

    TRAIN_FILE = FAMILY / "train" / TRAIN_NAME
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["split", "-b", f"{TRAIN_FILE.stat().st_size // pieces + 1}", str(TRAIN_FILE), f"{tmp}/part_"], check=True)
        parts = sorted(Path(tmp).iterdir())
        # sshd commonly limits concurrent unauthenticated connections (MaxStartups 10).
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            list(pool.map(lambda part: pod.scp_to(part, f"/root/chunks/{part.name}"), parts))
    target = f"{REMOTE_VOLUME}/data/double_gyre_shifting_wind_2layer/train/{TRAIN_FILE.name}"
    pod.ssh(f"cat /root/chunks/part_* > {target} && rm -rf /root/chunks && ls -l {target}")


def fetch_outputs(pod: Pod, destination: Path, checkpoints: bool) -> None:
    exclude = "" if checkpoints else "--exclude='*.pt' "
    tar = subprocess.run(["ssh", *pod.opts, "-p", pod.port, pod.host, f"tar czf - {exclude}--exclude='*.nc' -C {REMOTE_VOLUME}/outputs ."], check=True, capture_output=True)
    archive = destination / "outputs.tgz"
    archive.write_bytes(tar.stdout)
    with tarfile.open(archive) as handle:
        handle.extractall(destination / "outputs", filter="data")


def record(outputs: Path, checkpoints: bool) -> None:
    for result_path in sorted(outputs.glob("*/result.json")):
        output = json.loads(result_path.read_text())
        name = result_path.parent.name
        if output.get("returncode") != 0 or "experiment_log" not in output:
            log(f"FAILED {name}: {output.get('error') or output.get('stderr_tail', '')[-600:]}")
            continue
        for path in record_outputs(output, ROOT, FAMILY / "results.csv"):
            log(f"wrote {path}")
        checkpoint = result_path.parent / f"{name}.pt"
        if checkpoints and checkpoint.exists():
            (FAMILY / "checkpoints").mkdir(parents=True, exist_ok=True)
            checkpoint.replace(FAMILY / "checkpoints" / checkpoint.name)
            log(f"wrote checkpoint {checkpoint.name}")


def main() -> None:
    global ROOT, FAMILY
    parser = argparse.ArgumentParser(description="Run experiments on an hourly RunPod GPU pod")
    parser.add_argument("--job", action="append", required=True, help="<git ref>:<config path>")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--long-years", type=float, default=25.0)
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--gpu", action="append", default=None, help="CLOUD:gpu-id in preference order (repeatable); default: cheap community cards first")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    parser.add_argument("--keep", action="store_true", help="Do not delete the pod at the end")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Repository root whose data/ receives the results")
    args = parser.parse_args()
    ROOT, FAMILY = args.output_root, args.output_root / "data" / "double_gyre_shifting_wind_2layer"
    jobs = []
    for spec in args.job:
        ref, config = spec.split(":", 1)
        jobs.append({"ref": ref, "config": config, "diagnostics": args.diagnostics, "long_years": args.long_years})

    started = time.time()
    pod = create_pod(args.gpu or DEFAULT_GPUS)
    try:
        log(f"pod {pod.id} at {pod.host}:{pod.port}; setting up")
        pod.ssh(SETUP.format(volume=REMOTE_VOLUME))
        log("uploading training data")
        upload_data(pod)
        with tempfile.TemporaryDirectory() as tmp:
            jobs_file = Path(tmp) / "jobs.json"
            jobs_file.write_text(json.dumps(jobs))
            pod.scp_to(jobs_file, "/root/jobs.json")
        pod.ssh(
            f"cd /root/c_emulator && PATH=/root/venv/bin:$PATH nohup python -m src.remote.pod_worker /root/jobs.json "
            f"--volume {REMOTE_VOLUME} --work /root/work --concurrency {args.concurrency} > /root/queue.log 2>&1 &"
        )
        log(f"running {len(jobs)} job(s), {args.concurrency} at a time")
        deadline = time.time() + args.timeout_hours * 3600
        while pod.ssh(f"test -f {REMOTE_VOLUME}/DONE", check=False).returncode != 0:
            if time.time() > deadline:
                raise TimeoutError("queue did not finish in time")
            time.sleep(args.poll_seconds)
        log("queue finished; fetching outputs")
        with tempfile.TemporaryDirectory() as tmp:
            fetch_outputs(pod, Path(tmp), args.checkpoints)
            record(Path(tmp) / "outputs", args.checkpoints)
    finally:
        if args.keep:
            log(f"keeping pod {pod.id} (remember to delete it)")
        else:
            subprocess.run(["runpodctl", "pod", "delete", pod.id], capture_output=True)
            log(f"deleted pod {pod.id}")
    log(f"total wall time {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
