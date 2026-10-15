from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


RESULTS_CSV = Path("data/double_gyre_shifting_wind_2layer/results.csv")
LOG_DIR = Path("data/double_gyre_shifting_wind_2layer/logs")
CONFIG_ROOT = Path("configs")
EXPERIMENT_KINDS = {"tuning", "medium_step", "bigger_jump"}


def final_rollout_mean(log_path: Path) -> float | None:
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    rollouts = payload.get("rollouts", [])
    if not isinstance(rollouts, list):
        return None
    for rollout in reversed(rollouts):
        if not isinstance(rollout, dict):
            continue
        series = rollout.get("eval_mse_by_time")
        if isinstance(series, list) and series:
            return float(mean(float(value) for value in series))
    return None


def scheduled_rollout_summary(log_path: Path) -> list[tuple[int, float, float | None]]:
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    rows: list[tuple[int, float, float | None]] = []
    for rollout in payload.get("rollouts", []):
        if not isinstance(rollout, dict):
            continue
        series = rollout.get("eval_mse_by_time")
        if not isinstance(series, list) or not series:
            continue
        rows.append(
            (
                int(rollout["epoch"]),
                float(mean(float(value) for value in series)),
                None if rollout.get("eval_mse_last") is None else float(rollout["eval_mse_last"]),
            )
        )
    return rows


def results_rows(results_csv: Path = RESULTS_CSV) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paths_for_family(family: str) -> tuple[Path, Path]:
    """Return results CSV and log directory for an experiment family."""

    family_root = Path("data") / family
    return family_root / "results.csv", family_root / "logs"


def experiment_kind_from_config(config_path: Path) -> str:
    """Return the experiment-kind classification stored in a YAML config."""

    in_experiment_block = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("experiment:"):
            in_experiment_block = True
            continue
        if in_experiment_block and raw_line and not raw_line.startswith(" "):
            break
        if in_experiment_block:
            stripped = raw_line.strip()
            if stripped.startswith("kind:"):
                value = stripped.split(":", 1)[1].strip().strip('"\'')
                return value if value in EXPERIMENT_KINDS else "unknown"
    return "unknown"


def config_path_for_experiment(experiment_id: str, config_root: Path = CONFIG_ROOT) -> Path | None:
    """Find the YAML config matching an experiment id."""

    direct_matches = sorted(config_root.glob(f"*/{experiment_id}.yaml"))
    if direct_matches:
        return direct_matches[0]
    recursive_matches = sorted(config_root.glob(f"**/{experiment_id}.yaml"))
    return recursive_matches[0] if recursive_matches else None


def recent_experiment_kind_mix(log_dir: Path = LOG_DIR, config_root: Path = CONFIG_ROOT, limit: int = 8) -> dict[str, object]:
    """Summarise recent tuning/medium/bigger-jump mix from logs and configs."""

    kinds: list[str] = []
    for log_path in sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        experiment_id = str(metadata.get("experiment_id", log_path.stem))
        config_path = config_path_for_experiment(experiment_id, config_root)
        kinds.append("unknown" if config_path is None else experiment_kind_from_config(config_path))
    counts = {kind: kinds.count(kind) for kind in ["tuning", "medium_step", "bigger_jump", "unknown"]}
    streak_kind = kinds[0] if kinds else None
    streak_length = 0
    for kind in kinds:
        if kind != streak_kind:
            break
        streak_length += 1
    recommendation = "choose any kind based on evidence"
    if streak_kind == "tuning" and streak_length >= 2:
        recommendation = "next experiment should be medium_step or bigger_jump"
    if counts["bigger_jump"] == 0 and len(kinds) >= 4:
        recommendation = "next experiment should be bigger_jump unless recent evidence strongly justifies otherwise"
    return {"kinds": kinds, "counts": counts, "streak_kind": streak_kind, "streak_length": streak_length, "recommendation": recommendation}


def main() -> None:  # pragma: no cover - CLI reporting wrapper
    parser = argparse.ArgumentParser(description="Summarize ocean-emulator autoresearch status")
    parser.add_argument("--family", default="double_gyre_shifting_wind_2layer", help="experiment family under data/ and configs/")
    args = parser.parse_args()
    results_csv, log_dir = paths_for_family(args.family)

    print(f"Autoresearch status: {args.family}")
    print("====================" + "=" * len(args.family))
    rows = results_rows(results_csv)
    full_rows = [row for row in rows if row.get("epochs_completed") not in {"", "1", None}]
    if full_rows:
        best = min(full_rows, key=lambda row: float(row["eval_mse_mean"]))
        print("Best full-run CSV result:")
        print(f"  experiment_id: {best['experiment_id']}")
        print(f"  eval_mse_mean: {float(best['eval_mse_mean']):.6g}")
        print(f"  eval_mse_last: {float(best['eval_mse_last']):.6g}")
        print(f"  epochs_completed: {best['epochs_completed']}")
    else:
        print("No full-run CSV results found.")

    print("\nRecent experiment-kind mix:")
    mix = recent_experiment_kind_mix(log_dir=log_dir)
    print(f"  counts: {mix['counts']}")
    print(f"  current_streak: {mix['streak_kind']} x{mix['streak_length']}")
    print(f"  recommendation: {mix['recommendation']}")

    print("\nRecent logs:")
    for log_path in sorted(log_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:8]:
        final_mean = final_rollout_mean(log_path)
        print(f"  {log_path.name}: final_eval_mse={final_mean if final_mean is not None else 'missing'}")
        schedule = scheduled_rollout_summary(log_path)
        if schedule:
            formatted = ", ".join(f"e{epoch}:{mse:.3g}" for epoch, mse, _last in schedule)
            print(f"    schedule: {formatted}")


if __name__ == "__main__":  # pragma: no cover
    main()
