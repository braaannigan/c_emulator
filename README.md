# Ocean Model ML Emulator — dgsw2l

## Project goal

This repository supports a scientific research project to develop a
machine-learning emulator for an ocean model. The aim is to train ML models on
output from a numerical ocean model and use the trained emulator to reproduce
the model's dynamical evolution at substantially lower computational cost.

## Research objective

Ocean-model output is used as training data for ML models that learn the
time-evolution of the ocean state. The emulator is evaluated by autoregressive
rollout: it predicts the next ocean state, feeds that prediction back in as
input, and repeats over many time steps.

The central objective is **stable, long-horizon autoregressive rollouts that
remain faithful to the underlying ocean physics**. A successful emulator should:

- reproduce the key dynamical behaviour of the source ocean model;
- remain numerically stable over long rollouts;
- avoid unphysical drift or blow-up when run autoregressively;
- preserve physically meaningful ocean structures and variability;
- provide a useful surrogate for scientific analysis and experimentation.

## Scientific motivation

Numerical ocean models are essential for understanding ocean dynamics, climate
variability, and physical processes across scales, but they are expensive to
run, especially at high resolution or for large ensembles. A robust emulator
could accelerate experimentation while keeping the physically meaningful
behaviour of the original model.

## The system being emulated

Family `double_gyre_shifting_wind_2layer` (acronym `dgsw2l`): a two-layer
double-gyre simulation from the Aronnax ocean model, driven by a
zonal wind stress whose pattern shifts sinusoidally in latitude.

Training data: `data/double_gyre_shifting_wind_2layer/train/20260408T071500-period40d-duration5000d-2layer.nc`

- 715 snapshots, one every 7 days (5000 days total); model time step 600 s
- 2 layers on a 200 (y) × 100 (x) grid; C-grid staggered velocities
  (`xp1 = 101`, `yp1 = 201`)
- variables: `layer_thickness`, `zonal_velocity`, `meridional_velocity`
- wind: max stress 0.05, shift amplitude 40 km, shift period 40 days. Wind
  stress is not stored; `src/data.py` rebuilds it from the dataset attributes.

Preprocessing in the starter pipeline: velocities are averaged onto the
thickness grid; the state is `[h0, h1, u0, u1, v0, v1]` and the wind stress is
appended as a conditioning channel.

## Evaluation (fixed)

- Chronological split: first 80% of the record for training, last 20% held out.
- Every scheduled evaluation runs a **fully autoregressive rollout** across the
  held-out period through `src/evaluation.py`. The emulator gets only past truth
  and the known wind forcing; every predicted field must come from the model.
- Primary metric `eval_mse_mean`: mean over the rollout of layer-thickness MSE
  (m²). Also reported: `eval_mse_last` (final step) and `velocity_mse_mean`.
- The evaluator, split and data are the benchmark and must not be changed.

## Design principles (from the literature review)

Background synthesis lives in `docs/ocean_emulator_deep_research.md`; use it as
scientific context when proposing hypotheses. A recurring lesson is that ocean
emulator failures are often caused less by generic model capacity and more by an
underspecified forcing, memory, boundary, and physics interface.

- **Forcing interface first, architecture second.** Emulators are sensitive to
  how wind stress, seasonal phase and boundary inputs enter the model.
- **Treat staggered grids and boundaries as first-class modelling objects**
  rather than hiding them in preprocessing.
- **Evaluate scientific usability, not just MSE.** Lead-time error curves,
  spectral and variance diagnostics, kinetic energy, budget/integral checks and
  forcing-response tests all matter as the emulator matures.
- **Control drift and error accumulation explicitly**, and test candidate
  techniques as separate ablations rather than stacking them by default.
- **Stability is not the same as correct forced response.** A model can stay
  stable while responding wrongly to the forcing.
- **Seed and initial-condition robustness are core metrics.** Judge incumbents
  on consistently trained final models, not isolated lucky checkpoints.
- **Escalate architecture only after diagnosing the interface.** CNNs/U-Nets
  suit this gridded setting; graph, transformer or latent models are options
  when evidence points there.

## Experiment conventions

- Each experiment has an id `<YYYYMMDDTHHMMSS>-dgsw2l-<short-summary>`, a git
  branch of the same name, and a config in
  `configs/double_gyre_shifting_wind_2layer/<id>.yaml` with `experiment.kind`
  (`tuning` | `medium_step` | `bigger_jump`), `rationale`, `hypothesis`, and a
  `conclusion` written after the run.
- Test-driven development: write or update unit tests first, keep `src/` coverage
  at 100% where feasible, and all tests must pass before a run.
- Every experiment passes a 1-epoch smoke test before its full run. The default
  full run is 60 epochs, with a scheduled rollout every 5 epochs.
- Outputs: a live JSON log per run in `data/.../logs/`, a row in
  `data/.../results.csv`, and a final-step surface-thickness snapshot in
  `data/.../rollout/`. Full rollout NetCDFs are kept only for incumbents.
- Incumbent policy: a run that beats the current incumbent on `eval_mse_mean`
  with a consistent scheduled-rollout sequence (and survives a seed check) is
  merged to `main`. Record every experiment and the incumbent in `RESEARCH_LOG.md`.

## Commands

```bash
uv sync --extra dev
uv run --extra dev pytest --cov=src --cov-report=term-missing
uv run python -m src.pipelines.dgsw2l_baseline --config configs/double_gyre_shifting_wind_2layer/20261015T120000-dgsw2l-baseline-state-cnn.yaml --smoke-test
uv run python -m src.pipelines.dgsw2l_baseline --config configs/double_gyre_shifting_wind_2layer/20261015T120000-dgsw2l-baseline-state-cnn.yaml
uv run python scripts/autoresearch_status.py
uv run streamlit run src/viewer/training_metrics_viewer.py
```

In Claude Code, run `/autoresearch` to start the research loop.
