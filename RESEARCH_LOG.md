# dgsw2l research log

## Current incumbent

`20261015T120000-dgsw2l-baseline-state-cnn` (reference only — unstable,
final eval_mse_mean 1.97e5). Any stable model beats it.

## Reference scores (same evaluator window, 143 steps, no learning)

| reference | eval_mse_mean | eval_mse_last | velocity_mse_mean |
|---|---|---|---|
| persistence (repeat start state) | 297.7 | 471.0 | 0.0055 |
| training-mean climatology | 136.8 | 202.9 | 0.0021 |
| oracle eval-period mean (not achievable) | 125.2 | – | – |

Data facts: per-layer total volume is exactly conserved (mean h = 200 m in
both layers); spin-up ~ first 75 snapshots; thickness anomaly
autocorrelation 0.95 / 0.83 / 0.67 / 0.34 / -0.17 at lags 1/2/3/5/10 steps
(eddy-chaotic); a sin/cos regression on wind phase gives 133.8, i.e. the
forced response at the 7-day sampling is weak. Implication: beyond ~10 steps
a deterministic forecast cannot track the eddies; MSE rewards relaxing to the
climatological mean (~134 floor), while a variance-preserving free-running
emulator would asymptote near 2x the climate variance (~270, like
persistence). Track both MSE and variance/KE diagnostics.

## Experiments

### 20261015T120000-dgsw2l-baseline-state-cnn — bigger_jump (starter)
- Hypothesis: small 3-layer CNN, full next-state target, one-step MSE.
- Rollouts (epoch: mean / last / vel): 5: 8.5e21 / 3.5e23 / 1.7e15;
  10: 7.3e12 / 1.3e14 / 1.1e7; 15: 8.2e7 / 1.2e9 / 617; 20: 1.97e5 / 9.9e5 / 1.27.
- Conclusion: unstable in every rollout; error growth super-linear. Needs
  residual target, conservation, and multi-step training.
