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

### 20261015T160500-dgsw2l-residual-mass-cnn — medium_step
- Hypothesis: Predicting the normalised one-step increment (residual target) and projecting out the per-layer domain-mean thickness change (exact volume conservation) keeps the same small CNN stable over the 143-step held-out rollout and beats persistence (297.7).
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 6.287e+04 / 5.631e+05 / 0.5985 (std 4.67, KE 103.59); 10: 3.404e+05 / 4.45e+06 / 5.228 (std 8.83, KE 897.49); 15: 8.758e+07 / 1.796e+09 / 2521 (std 105.17, KE 424982.20); 20: 1.163e+10 / 3.095e+11 / 5.755e+05 (std 1034.10, KE 96770182.47); 25: 1.406e+13 / 4.755e+14 / 1.01e+09 (std 31104.81, KE 170009835744.94); 30: 1.222e+16 / 4.84e+17 / 9.534e+11 (std 814470.06, KE 160713524991492.16); 35: 4.275e+18 / 1.901e+20 / 3.558e+14 (std 14065370.00, KE 60073401886882560.00); 40: 1.112e+21 / 5.5e+22 / 1.039e+17 (std 211708432.00, KE 17566221657279152128.00); 45: 3.74e+22 / 1.958e+24 / 3.562e+18 (std 1183315328.00, KE 602757526139678490624.00); 50: 4.064e+24 / 2.264e+26 / 4.06e+20 (std 11891791872.00, KE 68764645935231912640512.00); 55: 2.767e+26 / 1.597e+28 / 2.749e+22 (std 95242633216.00, KE 4657870939975430583615488.00); 60: 3.056e+27 / 1.821e+29 / 3.019e+23 (std 311505256448.00, KE 51183170990709710740520960.00)
- Conclusion: Rejected: unstable. The residual target with exact layer-volume projection greatly improves short-lead skill (step-1 thickness MSE 4.2 vs 15 for the baseline; step-5 ~70 vs 143) but the one-step-trained model still diverges exponentially in every scheduled rollout (final 3.1e27), and divergence worsens with training. One-step MSE does not constrain the error-propagation dynamics; next test multi-step unrolled training on top of this recipe.

### 20261015T163000-dgsw2l-unroll8-residual-mass — medium_step
- Hypothesis: Training the residual mass-conserving CNN on 8-step autoregressive unrolls (backprop through its own predictions, loss averaged over steps, grad clip 1.0) makes the 143-step rollout stable and, because MSE over chaotic multi-week horizons rewards relaxation toward the mean, pushes eval_mse_mean below climatology (136.8).
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 564.3 / 1263 / 0.005006 (std 1.04, KE 0.43); 10: 698.4 / 2504 / 0.006823 (std 1.01, KE 0.64); 15: 2652 / 2.47e+04 / 0.02556 (std 1.22, KE 4.07); 20: 4.436e+04 / 5.75e+05 / 0.6561 (std 3.14, KE 112.15); 25: 9.797e+05 / 1.784e+07 / 30.74 (std 11.93, KE 5190.83); 30: 2.351e+07 / 5.727e+08 / 1428 (std 50.02, KE 240270.50); 35: 6.7e+07 / 1.749e+09 / 4301 (std 80.66, KE 723630.16); 40: 4.13e+08 / 1.175e+10 / 2.965e+04 (std 188.81, KE 4988359.61); 45: 5.672e+09 / 1.797e+11 / 4.835e+05 (std 653.48, KE 81358208.93); 50: 9.179e+10 / 3.101e+12 / 8.497e+06 (std 2528.04, KE 1430477168.39); 55: 2.949e+11 / 1.029e+13 / 2.712e+07 (std 4432.94, KE 4566778177.58); 60: 7.322e+11 / 2.604e+13 / 6.598e+07 (std 6882.54, KE 11113032939.55)
- Conclusion: Rejected: still unstable, but slower divergence. 8-step unrolled training improves mid-lead skill (step-10 thickness MSE ~70 vs ~158 for one-step training; step-5 ~65) and early rollouts stay near physical variance (epoch 5: h-std ratio 1.04, eval_mse_mean 564), but every scheduled rollout eventually diverges after ~20-50 steps and the divergence worsens as training proceeds (final 7.3e11). The model has no reference state to relax toward and no notion of position (walls, western boundary); next candidates: static climatology/coordinate context, noise injection, larger receptive field.
