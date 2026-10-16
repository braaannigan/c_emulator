"""Compare experiments on the benchmark and the secondary scorecard.

Usage:
    uv run python scripts/scorecard_table.py <experiment-id> [<experiment-id> ...]

Seed replicas (``<id>-seed<N>``) found in the logs are grouped with their
base id and averaged. Columns (lower is better for all):

- ``mse30-60`` / ``mse_final``: benchmark ``eval_mse_mean`` (epochs 30-60 average / final epoch)
- ``pred30-60``: predictable-range MSE (first 26 steps), epochs 30-60 average, from the logs
- ``realism`` / ``noise``: held-out scorecard of the final checkpoint (from the run's
  ``scorecard_final`` or its fidelity-diagnostics report)
- ``drift``: long-run drift score from the fidelity-diagnostics report
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scorecard import drift_score, predictable_mse, realism_from_statistics  # noqa: E402

FAMILY = Path(os.environ.get("DGSW2L_DATA", ROOT / "data" / "double_gyre_shifting_wind_2layer"))


def run_scores(experiment_id: str) -> dict[str, float]:
    log = json.loads((FAMILY / "logs" / f"{experiment_id}.json").read_text())
    rollouts = {r["epoch"]: r for r in log["rollouts"]}
    late = [r for e, r in rollouts.items() if 30 <= e <= 60]
    final = rollouts[max(rollouts)]
    scores = {
        "mse30-60": float(np.mean([r["eval_mse_mean"] for r in late])) if late else float("nan"),
        "mse_final": final["eval_mse_mean"],
        "pred30-60": float(np.mean([predictable_mse(r["eval_mse_by_time"]) for r in late])) if late else float("nan"),
    }
    card = log.get("scorecard_final")
    report_path = FAMILY / "diagnostics" / experiment_id / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    card = card or report.get("held_out", {}).get("scorecard")
    if not card and "statistics" in report.get("held_out", {}):
        card = realism_from_statistics(report["held_out"]["statistics"])
    scores["realism"] = card["realism"] if card else float("nan")
    scores["noise"] = card["noise_excess_decades"] if card else float("nan")
    long_run = report.get("long_run", {})
    if "drift" in long_run:
        scores["drift"] = long_run["drift"]["drift_score"]
    elif "yearly_blocks" in long_run:
        scores["drift"] = drift_score(long_run["yearly_blocks"]["kinetic_energy"], long_run["truth_yearly_blocks"]["kinetic_energy"])["drift_score"]
    else:
        scores["drift"] = float("nan")
    return scores


def main() -> None:
    columns = ["mse30-60", "mse_final", "pred30-60", "realism", "noise", "drift"]
    print(f"{'experiment':58s} {'seeds':>5s} " + " ".join(f"{c:>9s}" for c in columns))
    for base in sys.argv[1:]:
        ids = [base] + sorted(p.stem for p in (FAMILY / "logs").glob(f"{base}-seed*.json") if re.fullmatch(rf"{re.escape(base)}-seed\d+", p.stem))
        rows = [run_scores(i) for i in ids if (FAMILY / "logs" / f"{i}.json").exists()]
        if not rows:
            print(f"{base:58s}  (no logs)")
            continue
        means = {c: np.nanmean([r[c] for r in rows]) if any(np.isfinite(r[c]) for r in rows) else float("nan") for c in columns}
        print(f"{base:58s} {len(rows):5d} " + " ".join(f"{means[c]:9.3g}" for c in columns))


if __name__ == "__main__":
    main()
