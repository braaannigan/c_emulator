from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


RESULT_COLUMNS = [
    "last_update_time",
    "experiment_id",
    "eval_mse_mean",
    "eval_mse_last",
    "epochs_completed",
    "hypothesis",
    "conclusion",
]


def append_result(
    csv_path: str | Path,
    experiment_id: str,
    eval_mse_mean: float,
    eval_mse_last: float,
    epochs_completed: int,
    hypothesis: str,
    conclusion: str | None,
) -> None:
    """Append one experiment summary row, creating the CSV if needed."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "last_update_time": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "eval_mse_mean": eval_mse_mean,
                "eval_mse_last": eval_mse_last,
                "epochs_completed": epochs_completed,
                "hypothesis": hypothesis,
                "conclusion": conclusion or "",
            }
        )
