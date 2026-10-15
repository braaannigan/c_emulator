from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed access to the experiment YAML used by training pipelines."""

    raw: dict[str, Any]
    path: Path

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment"]["id"])

    @property
    def hypothesis(self) -> str:
        return str(self.raw["experiment"]["hypothesis"]).strip()

    @property
    def conclusion(self) -> str | None:
        value = self.raw["experiment"].get("conclusion")
        return None if value is None else str(value)

    @property
    def training_data(self) -> Path:
        return Path(self.raw["paths"]["training_data"])

    @property
    def rollout_dir(self) -> Path:
        return Path(self.raw["paths"]["rollout_dir"])

    @property
    def results_csv(self) -> Path:
        if "results_csv" in self.raw["paths"]:
            return Path(self.raw["paths"]["results_csv"])
        return Path(self.raw["results"]["csv_path"])

    @property
    def experiment_log(self) -> Path:
        value = self.raw["paths"].get("experiment_log")
        if value is not None:
            return Path(value)
        return self.results_csv.parent / "logs" / f"{self.experiment_id}.json"

    @property
    def epochs(self) -> int:
        return int(self.raw["runtime"].get("epochs", 60))

    @property
    def smoke_test_epochs(self) -> int:
        return int(self.raw["runtime"].get("smoke_test_epochs", 1))

    @property
    def seed(self) -> int:
        return int(self.raw["runtime"].get("seed", 42))

    @property
    def train_fraction(self) -> float:
        return float(self.raw["split"].get("train_fraction", 0.8))

    @property
    def batch_size(self) -> int:
        return int(self.raw.get("training", {}).get("batch_size", 8))

    @property
    def learning_rate(self) -> float:
        return float(self.raw.get("training", {}).get("learning_rate", 1e-3))

    @property
    def weight_decay(self) -> float:
        return float(self.raw.get("training", {}).get("weight_decay", 0.0))

    @property
    def gradient_clip_norm(self) -> float | None:
        value = self.raw.get("training", {}).get("gradient_clip_norm")
        return None if value is None else float(value)

    @property
    def lr_decay_milestones(self) -> list[int]:
        return [int(value) for value in self.raw.get("training", {}).get("lr_decay_milestones", [])]

    @property
    def lr_decay_factor(self) -> float:
        return float(self.raw.get("training", {}).get("lr_decay_factor", 1.0))

    @property
    def rollout_steps(self) -> int | None:
        value = self.raw.get("rollout", {}).get("steps")
        return None if value is None else int(value)

    @property
    def eval_interval_epochs(self) -> int:
        return int(self.raw.get("rollout", {}).get("eval_interval_epochs", 5))

    @property
    def save_rollout_outputs(self) -> bool:
        """Whether to persist full rollout arrays.

        Defaults to False so experiments remain reproducible from code, config,
        data, and scalar/per-timestep metrics without accumulating derived
        NetCDF rollout artifacts.
        """

        return bool(self.raw.get("rollout", {}).get("save_outputs", False))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must contain a YAML mapping")
    return ExperimentConfig(raw=raw, path=config_path)
