# c_emulator — dgsw2l ocean-emulator autoresearch

Goal: find an ML emulator that produces stable, accurate, long-horizon
autoregressive rollouts of a two-layer double-gyre ocean with a shifting wind
(`double_gyre_shifting_wind_2layer`, acronym `dgsw2l`).

Read `README.md` for the research objective, the system being emulated, the
fixed evaluation, design principles and experiment conventions. To run the
research loop, use the `autoresearch` skill (`/autoresearch`).

## Ground rules

- **Stay inside this directory.** Do not read, search, or list anything outside
  `c_emulator/` (sibling projects in the parent directory are off limits). All
  evidence must come from experiments you run here.
- **The benchmark is fixed.** Never modify `src/evaluation.py`, the
  chronological 0.8/0.2 train/eval split, or the training NetCDF. Every model must
  be scored by `evaluate_rollout`, which hands the emulator only past truth and
  the known wind forcing. Do not give the emulator any other access to
  eval-period truth (e.g. via closures, globals, or reading the data file inside
  `rollout`).
- Primary metric: `eval_mse_mean` (mean layer-thickness MSE over the full
  held-out rollout, lower is better). Also watch `eval_mse_last` and
  `velocity_mse_mean`, and judge models on the whole sequence of scheduled
  rollouts, not one lucky checkpoint.

## Layout

- `data/double_gyre_shifting_wind_2layer/train/20260408T071500-period40d-duration5000d-2layer.nc`
  — training data (Aronnax two-layer run: 715 snapshots every 7 days, 2 layers,
  200×100 grid, wind-shift period 40 days, amplitude 40 km).
- `configs/double_gyre_shifting_wind_2layer/` — one YAML per experiment.
- `data/double_gyre_shifting_wind_2layer/{logs,rollout,results.csv}` — outputs.
- `src/pipelines/dgsw2l_baseline.py` — training pipeline (plain CNN baseline).
- `src/evaluation.py` — fixed rollout evaluator (do not edit).
- `docs/ocean_emulator_deep_research.md` — literature review.
- `RESEARCH_LOG.md` — your running log of experiments and the current incumbent.

## Commands

```bash
uv run --extra dev pytest --cov=src --cov-report=term-missing
uv run python -m src.pipelines.dgsw2l_baseline --config <config> --smoke-test
uv run python -m src.pipelines.dgsw2l_baseline --config <config>
uv run python scripts/autoresearch_status.py
uv run python scripts/fidelity_diagnostics.py --config <config> [--long-years 25]   # spectra, EKE maps, decorrelation, long-run drift
uv run python scripts/runpod_run.py --config <config> [--ref <branch>]           # train on the RunPod endpoint (one job at a time)
uv run streamlit run src/viewer/training_metrics_viewer.py
```
