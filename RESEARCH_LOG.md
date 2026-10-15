# dgsw2l research log

## Current incumbent

`20261015T191000-dgsw2l-noise-injection` (seed check:
`20261015T200500-dgsw2l-noise-injection-seed7`). First stable emulator.

- Recipe: 3-hidden-layer 32-ch CNN, residual (increment) target, exact
  per-layer volume conservation over wet cells, land ring pinned, static
  inputs (train-mean state, coordinates, land mask), 8-step unrolled MSE,
  white input noise of 1 increment-std on each window start, Adam 3e-4,
  clip 1.0, 60 epochs.
- Final rollout: eval_mse_mean 215.7 / last 393.6 / vel 0.00258 (seed 7:
  209.3 / 307.6 / 0.00247). All 12 scheduled rollouts bounded (155-261).
- Variance-preserving (h-std ratio ~1.0, KE ratio ~0.78). Beats persistence
  (297.7); does not beat climatology (136.8). Error saturates near 2x
  climate variance after ~30 steps, which is what a faithful free-running
  emulator of a chaotic flow should do.
- Full rollout: `data/double_gyre_shifting_wind_2layer/rollout/20261015T191000-dgsw2l-noise-injection-final-epoch0060.nc`;
  checkpoint in `data/double_gyre_shifting_wind_2layer/checkpoints/`.

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

### 20261015T170500-dgsw2l-static-context — medium_step
- Hypothesis: Appending static context channels -- the training time-mean state (normalised like the state) and y/x coordinates in [-1, 1] -- to the unroll-8 residual mass-conserving CNN lets it learn location-dependent dynamics and relaxation toward the mean state, yielding a stable 143-step rollout with eval_mse_mean below climatology (136.8).
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 178.1 / 333.7 / 0.002319 (std 0.97, KE 0.68); 10: 242.5 / 643.2 / 0.002876 (std 1.01, KE 0.81); 15: 388.8 / 1128 / 0.00464 (std 0.96, KE 1.21); 20: 777 / 6405 / 0.02601 (std 1.06, KE 4.83); 25: 2877 / 3.568e+04 / 0.1374 (std 1.37, KE 23.94); 30: 9656 / 1.354e+05 / 0.4341 (std 1.85, KE 74.48); 35: 5.287e+04 / 8.504e+05 / 2.448 (std 3.35, KE 416.77); 40: 1.909e+05 / 3.361e+06 / 7.804 (std 5.76, KE 1324.10); 45: 8.59e+05 / 1.639e+07 / 35.01 (std 11.08, KE 5921.20); 50: 3.256e+06 / 6.746e+07 / 140.3 (std 20.45, KE 23669.53); 55: 1.313e+07 / 2.917e+08 / 586.1 (std 39.13, KE 98720.87); 60: 4.025e+07 / 9.37e+08 / 1830 (std 67.25, KE 308192.41)
- Conclusion: Rejected: still unstable, but clearly better early. Static context (training-mean state + coordinates) lowers the epoch-5 rollout to 178 (vs 564 without) with physical variance (h-std ratio 0.97, KE 0.68) and step-20 MSE ~100, yet divergence still sets in after ~60 steps and worsens with training (final 4.0e7). Checkpoint diagnosis: the blow-up is seeded at the western wall (x=0, y~186, the western-boundary-current region) and is grid-scale (>40% of error energy at |k|>0.25 by step 60), with h0 going negative. Points to the zero-padded boundary treatment as the instability source.

### 20261015T174500-dgsw2l-reflect-pad-boundary — bigger_jump
- Hypothesis: Replacing zero padding with reflect padding in every conv layer (a Neumann-like closed-wall boundary treatment, with the coordinate channels still marking wall position) removes the western-wall grid-scale instability of the static-context unroll-8 residual mass-conserving CNN, giving stable 143-step rollouts at every scheduled checkpoint.
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 178.7 / 341.1 / 0.002361 (std 0.98, KE 0.70); 10: 261.5 / 673.5 / 0.003022 (std 0.99, KE 0.86); 15: 473.7 / 1359 / 0.005124 (std 0.96, KE 1.31); 20: 982.4 / 8850 / 0.03747 (std 1.07, KE 6.81); 25: 5948 / 8.226e+04 / 0.3477 (std 1.62, KE 59.86); 30: 2.217e+04 / 3.355e+05 / 1.223 (std 2.43, KE 208.89); 35: 1.436e+05 / 2.479e+06 / 8.981 (std 5.05, KE 1524.11); 40: 5.443e+05 / 1.01e+07 / 33.47 (std 9.18, KE 5665.91); 45: 2.175e+06 / 4.388e+07 / 146.5 (std 17.06, KE 24726.33); 50: 8.656e+06 / 1.879e+08 / 692.6 (std 32.47, KE 116731.24); 55: 2.887e+07 / 6.647e+08 / 2688 (std 57.18, KE 452534.27); 60: 7.214e+07 / 1.74e+09 / 8183 (std 88.73, KE 1376928.83)
- Conclusion: Rejected: no effect. Reflect padding gives rollouts nearly identical to zero padding (epoch 5: 178.7 vs 178.1; final 7.2e7 vs 4.0e7) and the blow-up is still seeded at x=0, y~186. Follow-up data inspection found the real boundary structure: the outer ring (rows 0,1,198,199 and alternate rows of columns 0 and 99, 596 cells) is land, with h=200 and u=v=0 fixed for all time, in a checkerboard along the side walls. The residual model lets these cells drift, and the side-wall checkerboard is exactly where the grid-scale instability starts. Next: hard land mask.

### 20261015T183000-dgsw2l-land-mask — bigger_jump
- Hypothesis: Treating land as a first-class grid object -- pinning the 596 cells that are constant over the training split to their fixed values after every step, projecting the volume correction over wet cells only, and feeding the land mask as an input channel -- removes the side-wall instability of the static-context unroll-8 residual mass-conserving CNN, yielding stable 143-step rollouts at every scheduled checkpoint with eval_mse_mean below climatology (136.8).
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 164.2 / 290.6 / 0.002226 (std 0.94, KE 0.72); 10: 236.8 / 633.4 / 0.003054 (std 0.97, KE 0.86); 15: 311.4 / 934.7 / 0.004629 (std 1.01, KE 1.18); 20: 317.9 / 886 / 0.004549 (std 1.06, KE 1.16); 25: 371.6 / 1485 / 0.006063 (std 0.96, KE 1.39); 30: 623.8 / 5875 / 0.0187 (std 1.08, KE 3.55); 35: 2593 / 3.603e+04 / 0.1012 (std 1.33, KE 17.59); 40: 2.582e+04 / 4.212e+05 / 1.067 (std 2.51, KE 181.82); 45: 2.24e+05 / 4.225e+06 / 10.12 (std 5.84, KE 1711.59); 50: 1.233e+06 / 2.565e+07 / 55.76 (std 12.52, KE 9407.10); 55: 1.025e+07 / 2.395e+08 / 530.7 (std 33.06, KE 89341.83); 60: 8.522e+07 / 2.224e+09 / 4964 (std 89.61, KE 834916.80)
- Conclusion: Not adopted as incumbent (unstable at the end of training) but adopted as the base recipe. Pinning the land ring and wet-cell volume projection gives the best early rollouts so far (epoch 5: 164.2, the only rollout below persistence besides static-context's, and h-std ratio 0.94) and stays bounded through epoch 25 (311-372), roughly doubling the number of stable scheduled rollouts, but divergence still develops from epoch 30 (final 8.5e7). Diagnosis: the blow-up has moved off the walls into the interior where the western boundary current separates (y~182-191, x~9-13) and is grid-scale (79% of error energy at |k|>0.25 by step 100). Next: teach the model to damp its own grid-scale errors (noise injection).

### 20261015T191000-dgsw2l-noise-injection — medium_step
- Hypothesis: Adding white noise (1 increment-std per channel, wet cells only) to the start state of every 8-step training window, with clean true targets, teaches the land-mask CNN to damp grid-scale errors and keeps its 143-step rollouts bounded at every scheduled checkpoint, with eval_mse_mean below climatology (136.8).
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 154.9 / 263.9 / 0.002116 (std 0.94, KE 0.71); 10: 189.7 / 332.7 / 0.002094 (std 0.97, KE 0.68); 15: 226.6 / 401.7 / 0.002279 (std 0.99, KE 0.75); 20: 220.3 / 360.8 / 0.002348 (std 1.03, KE 0.76); 25: 261.2 / 426.8 / 0.002413 (std 0.92, KE 0.76); 30: 201.4 / 315.7 / 0.002536 (std 0.97, KE 0.82); 35: 194 / 288.2 / 0.002549 (std 0.97, KE 0.74); 40: 211.7 / 356.4 / 0.002605 (std 0.95, KE 0.74); 45: 209 / 317.8 / 0.002457 (std 0.98, KE 0.79); 50: 230.7 / 386.4 / 0.002537 (std 1.04, KE 0.86); 55: 246.8 / 364.3 / 0.002879 (std 1.00, KE 0.87); 60: 215.7 / 393.6 / 0.002584 (std 1.03, KE 0.78)
- Conclusion: Supported: first stable emulator. Noise injection (1 increment-std white noise on wet cells of each window's start state) on top of the land-mask recipe keeps all 12 scheduled 143-step rollouts bounded (eval_mse_mean 155-261, final 215.7 / last 393.6 / vel 0.00258), beating persistence (297.7) at every checkpoint, with physically consistent variance (h-std ratio 0.92-1.04, KE ratio 0.68-0.87, no late-lead growth). It does not beat climatology (136.8): it behaves as a variance-preserving free-running model whose error saturates near 2x climate variance after ~30 steps (~200-300), as anticipated by the predictability diagnostics. Short-lead skill is preserved (step-1 7.8, step-5 54, step-10 60). Candidate incumbent pending seed check.

### 20261015T200500-dgsw2l-noise-injection-seed7 — tuning
- Hypothesis: The noise-injection recipe is seed-robust: with seed 7 all scheduled rollouts stay bounded and the final eval_mse_mean is within ~20% of seed 42 (215.7).
- Rollouts (epoch: mean / last / vel (h-std ratio, KE ratio)): 5: 155.3 / 232.4 / 0.002142 (std 0.90, KE 0.73); 10: 192.8 / 419.9 / 0.002256 (std 0.90, KE 0.72); 15: 193.1 / 264 / 0.002219 (std 1.01, KE 0.69); 20: 182.5 / 277.8 / 0.002228 (std 0.96, KE 0.75); 25: 218.2 / 404.5 / 0.00243 (std 0.99, KE 0.74); 30: 199.4 / 319.7 / 0.002348 (std 1.01, KE 0.74); 35: 187.6 / 272.6 / 0.002359 (std 0.99, KE 0.72); 40: 202.7 / 352.9 / 0.002405 (std 0.99, KE 0.73); 45: 233.7 / 374.1 / 0.002315 (std 1.02, KE 0.74); 50: 224.6 / 384 / 0.002363 (std 1.04, KE 0.72); 55: 225.3 / 348.4 / 0.002254 (std 1.09, KE 0.80); 60: 209.3 / 307.6 / 0.002471 (std 1.01, KE 0.89)
- Conclusion: Seed check passed. Seed 7 keeps all 12 scheduled rollouts bounded (155-234), final 209.3 / last 307.6 / vel 0.00247 vs seed 42's 215.7 / 393.6 / 0.00258, with matching variance diagnostics (h-std ratio 0.90-1.09, KE 0.69-0.89). The noise-injection recipe is promoted to incumbent (seed-42 run as the reference; two-seed final mean 212.5).
