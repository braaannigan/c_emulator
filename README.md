# c_emulator

Fresh autoresearch project for the two-layer double-gyre shifting-wind ocean
emulator (`dgsw2l`). Start a Claude Code session in this directory and run
`/autoresearch`. See `CLAUDE.md` for the rules and layout.

```bash
uv sync --extra dev
uv run pytest
uv run python -m src.pipelines.dgsw2l_baseline --config configs/double_gyre_shifting_wind_2layer/20261015T120000-dgsw2l-baseline-state-cnn.yaml --smoke-test
```
