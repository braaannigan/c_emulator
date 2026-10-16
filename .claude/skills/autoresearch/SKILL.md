---
name: autoresearch
description: Run the dgsw2l ocean-emulator autoresearch loop. Use when asked to start, continue, or keep going with autoresearch experiments, hypotheses, branches, configs, tests, smoke tests, full runs, result write-up, and incumbent tracking.
---

# dgsw2l Autoresearch

Run a persistent research loop that develops ML emulators for
`double_gyre_shifting_wind_2layer` (`dgsw2l`) with stable, faithful,
long-horizon autoregressive rollouts. After each experiment, immediately begin
the next cycle unless the user explicitly says to stop or a hard tool/runtime
failure actually occurs. Do not stop because a future cycle might exceed time,
token, or tool-call budgets; running out mid-iteration is acceptable.

Additional user instructions: $ARGUMENTS

Read `CLAUDE.md` ground rules first: stay inside this directory and never alter
the fixed evaluator, split, or data.

## Loop

1. **Evaluate current evidence**
   - Read `data/double_gyre_shifting_wind_2layer/results.csv`, recent JSON logs in
     `data/double_gyre_shifting_wind_2layer/logs/`, and `RESEARCH_LOG.md`.
   - Run `uv run python scripts/autoresearch_status.py`.
   - Compare full scheduled-rollout sequences, not a single number. Identify the
     incumbent as the best *consistently trained* model. If no results exist,
     run the starter baseline config first.
   - Use `uv run python scripts/scorecard_table.py <ids>` to compare the
     benchmark (`eval_mse_mean`) with the secondary scorecard
     (`predictable_mse`, `realism`, `drift`), grouped over seeds, and apply
     the promotion rule in `CLAUDE.md`.

2. **Choose experiment kind, generate candidates, select one**
   - Classify every experiment:
     - `tuning`: hyperparameters, schedules, seeds, batch size, epochs, regularisation weights around the current recipe.
     - `medium_step`: changed objective, training horizon, forcing representation, or physically motivated loss with the broad architecture kept.
     - `bigger_jump`: changed architecture, conservation/projection mechanism, latent representation, grid/boundary treatment, or physical formulation.
   - Keep a deliberate mix: no more than 2 consecutive `tuning` runs unless one
     gave a new robust incumbent; after 2 tuning failures choose `medium_step` or
     `bigger_jump`; after 4 experiments without a new incumbent choose
     `bigger_jump`; follow a promising `bigger_jump` with a tuning or seed check.
   - Write 3–5 candidate hypotheses (mechanism, code/config changes, tests,
     acceptance criteria, failure modes) and pick the one that best balances
     expected rollout improvement, physical fidelity, seed robustness,
     complexity, and feasibility.

3. **Experiment id and branch**: `<YYYYMMDDTHHMMSS>-dgsw2l-<short-summary>`;
   `git switch -c <experiment-id>` from the incumbent's branch (or `main`).

4. **Config**: add `configs/double_gyre_shifting_wind_2layer/<experiment-id>.yaml`
   with hypothesis, paths, runtime, split, training, model, rollout, and
   `experiment.kind` + `experiment.rationale`. Default full run 60 epochs,
   smoke test 1 epoch, scheduled rollouts every 5 epochs. Keep
   `rollout.save_outputs: false` except for a new incumbent, whose final full
   rollout NetCDF should be retained.

5. **Implement test-first**: add/update unit tests for every changed code path;
   keep source coverage at 100% where feasible. Any new model must implement the
   `Emulator.rollout` interface and be scored through `src/evaluation.py`.

6. **Validate**
   ```bash
   uv run --extra dev pytest --cov=src --cov-report=term-missing
   uv run python -m src.pipelines.dgsw2l_baseline --config <config> --smoke-test
   ```
   The 1-epoch smoke run is the only required validation run. Check it actually
   exercised the new code path. A failing candidate is recorded as such and the
   loop moves on to a new hypothesis.

7. **Run** the full experiment:
   ```bash
   uv run python -m src.pipelines.dgsw2l_baseline --config <config>
   ```
   Long runs can go in the background; wait for them to finish rather than
   polling repeatedly.

8. **Analyse and record**
   - Report each scheduled rollout's `eval_mse_mean`, `eval_mse_last`, `velocity_mse_mean`,
     and the scorecard (`predictable_mse`, `realism`); for candidates near the
     incumbent run `scripts/fidelity_diagnostics.py` (or RunPod `--diagnostics`)
     for `drift`.
   - Write the conclusion into the config's `experiment.conclusion`.
   - Append the experiment to `RESEARCH_LOG.md` (id, kind, hypothesis, metric
     sequence, conclusion) and update the incumbent section.
   - Commit on the experiment branch. If it is a new incumbent, merge it to `main`.
   - Never promote a lucky intermediate checkpoint to incumbent.

9. **Continue**: start the next cycle immediately. Do not send per-experiment
   summaries during a multi-cycle run — the record lives in configs and
   `RESEARCH_LOG.md`. If an optional diagnostic fails, note it and carry on.

## Final response (only on stop/failure/limit)

Experiment ids, kinds and hypotheses, tests and smoke runs, scheduled rollout
metrics, conclusions, incumbent status, artifact paths, and why the loop stopped.
