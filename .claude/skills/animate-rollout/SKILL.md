---
name: animate-rollout
description: Render an MP4 animation of a trained dgsw2l emulator's rollout - the held-out free run next to the truth, or a long multi-year free run beyond the record - showing thickness anomaly, relative vorticity and a KE trace against the truth's range. Use when asked to animate, visualise or make a movie/video of a rollout, free run or model.
---

# Animate an emulator rollout

Renders rollouts straight from a trained checkpoint (no saved rollout NetCDF
needed), locally and inference-only (no RunPod cost).

## 1. Pick the model

- Default: the current incumbent. Find its experiment id in the
  "Current incumbent" section of `RESEARCH_LOG.md`; its config is
  `configs/double_gyre_shifting_wind_2layer/<id>.yaml`.
- The checkpoint must exist at
  `data/double_gyre_shifting_wind_2layer/checkpoints/<id>.pt`. Runs trained
  on RunPod keep checkpoints on the network volume under
  `outputs/<id>/<id>.pt`; if it is missing locally, fetch it with a
  short-lived CPU pod in the volume's datacenter (see
  `deploy/runpod/endpoint.json`) and `scp`, then delete the pod.
- Only grid-space stepper models (`model.type` `unet` / `state_cnn`) are
  supported; rejected latent models are not on `main`.

## 2. Render

```bash
# Held-out period: truth vs emulator (143 weekly frames, ~15 s)
uv run python scripts/animate_emulator.py --config <config> --mode held-out

# Long free run from the end of the record, analytic wind (default 25 years, ~30 s)
uv run python scripts/animate_emulator.py --config <config> --mode long --long-years 25
```

Options: `--layer 0|1` (upper/lower layer, default 0), `--stride N` (render
every Nth week; defaults 1 held-out / 2 long), `--fps N` (defaults 8 / 20),
`--checkpoint <pt>`, `--seed N` (stochastic models draw fresh noise each
run), `--output <path>`. Default output:
`figs/<id>-<mode>-layer<L>.mp4` (`figs/` is gitignored).

## 3. Check and report

- Extract a frame to sanity-check the layout before reporting, e.g.
  `ffmpeg -loglevel error -y -ss 10 -i <mp4> -frames:v 1 <scratchpad>/frame.png`
  and view it.
- Grey cells are land (thickness) or the land-mixed wall rows/columns
  (vorticity). Visible streaks along the walls in the emulator's thickness
  panel are a known model defect, not a rendering bug.
- Tell the user the MP4 path(s) and what to look for (eddy shedding at the
  western-boundary-current separation, KE trace staying inside the grey
  truth band, drift late in long runs). They can open it with
  `open <path>` (macOS).
- For quantitative fidelity (spectra, EKE maps, decorrelation, drift) use
  `scripts/fidelity_diagnostics.py` instead.

The older `scripts/render_rollout_mp4.py` animates a saved rollout NetCDF
(surface thickness only); prefer `animate_emulator.py`.
