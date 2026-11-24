# Experiments

## Registry

| run_id | question | candidate / config | code version | data split | command | output path | status | metrics | outcome | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| `covmatch-offline-20261123` | Can a constant mass-weighted covariance transform fix KE variance without breaking ACF/GK? | covariance-matching readout, molecular modes | `code/h1h2/test_covariance_match.py` | model: deterministic val_04 2.5 ns, first 100 ps; ref: NVT 2.5 ns, first 100 ps | `python test_covariance_match.py --model ... --reference ... --frames 20000 --stride-fs 5` | `experiments/covariance_match_20261123_v2.json` | completed | covariance rel. error ~1e-11; KE std 0.105 vs ref 0.240; GK 1.445 vs ref 1.502 (100 ps) | unsupported | reject Candidate 1; move to temporal kernel/noise candidate |
| `gle-interface-20261123` | Can a 5 fs displacement define a reliable instantaneous acceleration for a self-consistent GLE? | true NVE val_04: finite-difference acceleration vs \(F/m\) | analysis snippet | val_04 50 ps, 5 fs stride | finite-difference acceleration / impulse comparison | inline analysis | completed | \(a_{\rm fd}\) vs \(a_t\) corr 0.54, RMSE/std 0.87; \(\Delta v\) vs impulse corr 0.036 | unsupported / high-risk | drop 5 fs position-consistent GLE; use velocity-only GLE readout |
| `slow-stoch-100ps-20261123` | Does slow-mode + stochastic group rescale satisfy the water frontier gate? | GLR AR(10), 0.57% τ=200 fs slow mode, stochastic group rescale | `code/h1h2/glr_readout.py` | water val_04, 100 ps | rollout with `--glr-slow-tau 200 --glr-slow-weight 0.005706 --fd-stochastic-rescale` | `experiments/glr_slow_stoch_100ps_20261123` | completed | KE std 0.4405; D 1.480 (21.4% low); GK 1.386 vs MSD 1.480; O–H 0.1253; VDOS 0.858/0.970 | near-miss / unsupported | do not promote; slow mode is water-specific; proceed to velocity-only GLE |
| `velocity-gle-fit-20261123` | Can 3 damped oscillators + slow mode fit the translation ACF with the D constraint? | continuous PSD ACF fit | `code/h1h2/fit_velocity_gle.py` | reference NVT 100 ps, 5 fs | `fit_velocity_gle.py --modes 3` | `experiments/velocity_gle_fit_20261123.json` | completed | integral 11.29 vs 11.34 fs; ACF RMSE 0.0129; 200 fs −0.041 vs −0.080; 500 fs +0.006 vs −0.022 | unsupported / limited | 3 modes insufficient; do not implement; test AR(10)+slow no-rescale |
| `slow-norescale-100ps-20261123` | Does AR(10)+slow, without post-hoc rescale, hold the composed-system temperature? | GLR AR(10), slow mode, no rescale | `code/h1h2/glr_readout.py` | water val_04 | rollout with `--no-fd-rescale` | `experiments/glr_slow_norescale_100ps_20261123` | aborted at 1k | T=70.1 K | unsupported | local FD noise cannot control the composed system; need explicit variance control or stronger coupling |
| `slow-global-100ps-20261123` | Does deterministic group mean + global CSVR satisfy the water frontier gate? | GLR AR(10), slow mode, deterministic group rescale, global CSVR | `code/h1h2/fd_readout.py` | water val_04 | rollout with `--fd-rescale --fd-global-rescale --glr-slow-tau 200` | `experiments/glr_slow_global_100ps_20261123` | completed | total KE std 0.434; mode KE std 0.146/0.155/0.148 vs 0.239/0.234/0.250; D 1.246; GK 1.650; O–H 0.105; VDOS 0.855/0.973 | unsupported | global CSVR too coarse; test weak-coupling stochastic group τ=100 fs |
| `slow-weak-100ps-20261123` | Does weak-coupling stochastic group rescale τ=100 satisfy the water frontier gate? | slow mode + stochastic group rescale τ=100 | `code/h1h2/fd_readout.py` | water val_04 | rollout with `--fd-rescale-tau 100` | `experiments/glr_slow_weak_100ps_20261123` | **invalid** | KE std 0.0397/0.0616/0.0600; total 0.0708 | implementation error | old EMA target shrank Gamma variance; fix to Gamma AR(1) and rerun |
| `slow-weak-ar1-100ps-20261123` | Does corrected Gamma AR(1) τ=100 satisfy the water frontier gate? | slow mode + Gamma AR(1) group rescale τ=100 | `code/h1h2/fd_readout.py` | water val_04 | rollout with `--fd-rescale-tau 100` after fix | `experiments/glr_slow_weak_ar1_100ps_20261123` | completed | total KE std 0.318; mode KE std 0.187/0.191/0.190; D 1.361; GK 1.530; O–H 0.1229; VDOS 0.857/0.973 | unsupported | rescale family cannot satisfy mode variance + D/GK; structural redesign |
| `interface-resolution-20261123` | At what horizon is finite-difference acceleration a reliable drift? | true NVE val_04, 1/2/5 fs strides | analysis snippet | water val_04 | FD acceleration / impulse vs force | inline analysis | completed | 1 fs corr 0.987/0.948; 2 fs 0.920/0.798; 5 fs 0.541/0.036 | 1 fs positive; 2 fs marginal; 5 fs negative | Gate B: consider 1 fs backbone + position-consistent GLE |
| `water-1fs-backbone-20261123` | Train a 1 fs relational backbone for a position-consistent GLE | DSRT radius 6, 2.36M, horizon 1 fs, 250k steps | `code/h1h2/train_relational.py` | water 1 fs transitions 5000/1250/1000 | `train_relational.py --run experiments/water6_1fs_20261123` | `experiments/water6_1fs_20261123` | completed | test pos RMSE 3.16e-5 Å; vel RMSE 4.58e-5 Å/fs; epoch 393 | supported | backbone passes; proceed to 1 fs GLE kernel/integrator |
| `water-1fs-position-consistent-smoke-20261124` | Does the 1 fs blend position-consistent rollout hold T? | 1 fs backbone + AR(50) blend 0.1 | `position_consistent.py` | water val_04, 1 ps | `--position-consistent --fd-blend 0.1` | `experiments/water1fs_gle_smoke_20261124` | completed | T 70.9; KE std 1.10 | negative | local AR noise insufficient at blend 0.1 |
| `water-1fs-driven-gle-smoke-20261124` | Does AR(50)+backbone drift form a stable GLE? | driven AR GLE | `driven_gle.py` | water val_04, 1 ps | `--driven-gle` | `experiments/water1fs_driven_gle_smoke_20261124` | failed at 4 steps | preT 3509 K; min 0.50 Å | negative / invalid | AR predictor is not a passive GLE memory kernel |
| `water-1fs-langevin-smoke-20261124` | Is the 1 fs Langevin drift stable? | Langevin gamma 0.1 | `langevin_drift.py` | water val_04, 1 ps | `--langevin-drift --langevin-gamma 0.1` | `experiments/water1fs_langevin_smoke_20261124` | completed | T 256.8; KE std 0.395 | stable but cold | add global CSVR |
| `water-1fs-langevin-csVR-100ps-20261124` | Does Langevin + global CSVR satisfy the water frontier gate? | 1 fs backbone + Langevin drift + global CSVR | `langevin_drift.py` | water val_04, 100 ps | `--langevin-drift --langevin-gamma 0.1 --langevin-global-rescale` | `experiments/water1fs_langevin_csVR_100ps_20261124` | completed | T 301.47; partition 3.04/2.90/1.50; RDF O–H 0.862; VDOS 0.520/0.442; D 4.13e-10; GK 4.72e-10 | unsupported | integrator destroys partition/structure; stop this path |
| `velocity-gle-fit-1fs-20261124` | Can a 4–5 mode PSD kernel fit the 1 fs ACF with D? | damped oscillators + slow mode | `code/h1h2/fit_velocity_gle.py` | reference NVT, 20 ps, 1 fs | `fit_velocity_gle.py --modes 4/5` | `experiments/velocity_gle_fit_1fs_m4_20261124.json`, `...m5...` | completed | 5 fs ACF 0.315/0.346 vs 0.981; RMSE 0.071/0.066 | unsupported | PSD fit fails at 1 fs; stop integrator variants |

## Run detail

### covmatch-offline-20261123

- Question: can a constant mass-weighted covariance transform fix KE variance without breaking ACF/GK?
- Candidate/configuration: \(v' = A v\), \(A=C_{\rm ref}^{1/2}C_{\rm model}^{-1/2}\), molecular translation/rotation/internal.
- Comparator: deterministic GLR trajectory; reference NVT.
- Data: deterministic val_04 2.5 ns (first 100 ps); reference NVT 2.5 ns (first 100 ps), 5 fs stride.
- Budget: CPU only.
- Pre-registered: support if KE std and ACF/GK move toward reference; contradict if KE variance or ACF/GK remains off.
- Command: `python test_covariance_match.py --model ... --reference ... --frames 20000 --stride-fs 5 --maxlag 100`
- Code version: `code/h1h2/test_covariance_match.py`
- Output: `experiments/covariance_match_20261123_v2.json`
- Status: completed.
- Metrics:
  - covariance relative error after transform: ~1e-11;
  - translation KE std: model 0.00077, transformed 0.105, reference 0.240;
  - rotation KE std: 0.0435, 0.106, reference 0.238;
  - internal KE std: 0.0434, 0.186, reference 0.249;
  - translation ACF (5/25/100/500 fs): transformed 0.980/0.626/0.005/−0.020, reference 0.981/0.653/0.031/−0.022;
  - GK D: model 1.420, transformed 1.445, reference (100 ps) 1.502.
- Outcome: unsupported. Static covariance matching does not restore the temporal KE variance or D.
- Decision: reject Candidate 1; next candidate must add temporal memory/noise or a position-consistent mechanism.

### slow-stoch-100ps-20261123

- Question: does slow-mode + stochastic group rescale satisfy the water frontier gate?
- Configuration: GLR AR(10), 0.57% τ=200 fs translation slow mode, stochastic group rescale, 100 ps.
- Result: T=300.16, total KE std 0.4405 (ref 0.4417); partition 2.483/2.529/2.399; VDOS H/O 0.858/0.970; O–H RDF 0.1253; MSD D 1.480 (21.4% low); GK 1.386 (6.3% below MSD).
- Outcome: near-miss / unsupported. D is just outside the 20% gate; O–H is above the 0.118 frontier baseline; the slow mode is a water-specific component and has no quartz counterpart, so it is not a single rule.
- Decision: do not promote. Continue with the velocity-only GLE readout, which can be defined in the same physics basis for both materials.

### slow-norescale-100ps-20261123

- Question: does AR(10)+slow, without post-hoc rescale, hold the composed-system temperature?
- Configuration: GLR AR(10), 0.57% τ=200 fs slow mode, `--no-fd-rescale`, blend 0.1.
- Result: at 1k steps, T=70.1 K. The run was stopped; it cannot reach the 2.5 ns gate.
- Outcome: unsupported. The local FD-consistent AR process does not control the composed system's stationary variance; the 0.9 backbone weight dominates and cools.
- Decision: the readout needs explicit composed-system variance control (e.g., global CSVR) or a stronger coupling; a local kernel alone is insufficient.

### slow-weak-100ps-20261123 (invalid)

- Question: does weak-coupling stochastic group rescale τ=100 satisfy the water frontier gate?
- Configuration: slow mode + stochastic group rescale with `--fd-rescale-tau 100`.
- Observed: T=299.83, total KE std 0.0708; translation/rotation/internal KE std 0.0397/0.0616/0.0600; O–H 0.1231; VDOS 0.853/0.974; D 1.384; GK 1.661.
- **Implementation error:** the old `rescale_tau` branch used an exponential moving average of independent Gamma samples. Its stationary variance is `(1-alpha)/(1+alpha)` of the Gamma variance, which is ~2.6% at τ=100. The run did not test the intended weak-coupling scheme.
- Fix: Gamma AR(1), `K_{t+1}=alpha*K_t + Gamma(a(1-alpha), b)`, which preserves the Gamma(a,b) marginal.
- Status: invalid; rerun after the fix.

### water-1fs-backbone-20261123 (completed)

- Test: 1000 held-out 1 fs transitions.
- Position RMSE 3.16e-5 Å, velocity RMSE 4.58e-5 Å/fs; selected epoch 393, 245,625 optimizer steps; 13.5 h.
- The 1 fs backbone is accurate enough to supply a drift for the position-consistent GLE.
- Next: fit the 1 fs velocity ACF/AR/GLE kernel; implement the position-consistent integrator; 100 ps gate.

### velocity-gle-fit-1fs-20261124

- Question: can a 4–5 mode damped-oscillator + slow-mode PSD kernel fit the 1 fs reference ACF with the integral constraint?
- Result: 4 modes ACF@5 fs 0.315 vs 0.981, ACF@25 fs 0.228 vs 0.654, RMSE 0.071; 5 modes 0.346/0.257, RMSE 0.066.
- Outcome: the short-time ACF is not fit. A post-hoc PSD kernel at 1 fs is not a viable representation for the composed system.
- Decision: stop adding integrator/kernel variants; the drift and integrator must be trained together, or the objective must change.
