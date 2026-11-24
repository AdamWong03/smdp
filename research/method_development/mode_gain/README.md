# Mode-Gain Diagnostic — 100 ps NVT

**Date:** 2026-11-21
**Scripts:** `code/h1h2/diagnose_mode_gain.py`, `code/h1h2/diagnose_one_step_gain.py`
**Raw outputs:** `*.json` in this directory.

---

## 1. Trajectory-level diagnostic

Protocol: 50 ps thermalization + 100 ps production, CSVR 300 K, τ = 50 fs, 5 fs horizon,
val_04 frame 0. Reference: 100 ps block of the 2.5 ns SPC/Fw NVT trajectory, sampled at 5 fs.

Mode kinetic energy (eV, 64 molecules; equipartition ≈2.48 eV per mode) and mass-weighted
velocity autocorrelation α(τ):

| Model | trans KE | rot KE | int KE | total | α_t(25 fs) | α_r(25 fs) | α_i(5 fs) | α_i(100 fs) | α_i(500 fs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 2.373 | 2.430 | 2.609 | 7.411 | 0.658 | −0.540 | −0.567 | 0.147 | 0.104 |
| 250k Full | 0.538 | 0.786 | 6.044 | 7.368 | 0.393 | −0.783 | −0.902 | 0.831 | 0.722 |
| Current-Only | 0.646 | 0.535 | 6.384 | 7.565 | 0.646 | −0.764 | −0.917 | 0.858 | 0.590 |
| Native ep145 | 3.256 | 3.105 | 1.041 | 7.402 | 0.697 | −0.412 | +0.109 | −0.288 | 0.003 |

Total kinetic energy is approximately correct in every model (CSVR enforces it); the partition
is not.

- Our models: collective modes cold, internal modes hot and **underdamped** (internal ACF is
  still 0.72 at 500 fs; reference 0.10).
- Native ep145: collective modes hot, internal modes cold and **overdamped**.

The script's per-molecule values agree with the official partition values above to ~2%. The
small difference between the sum of modes and the total is due to the instantaneous-inertia
rotation/internal decomposition not being exactly orthogonal for flexible molecules.

## 2. One-step diagnostic on held-out NVE pairs

250k Full, 1,250 validation graphs, mass-weighted mode statistics:

| Mode | α_ref | α_pred | Var ratio | residual / reference variance |
|---|---:|---:|---:|---:|
| translation | 0.98133 | 0.98154 | 1.00044 | 2.1e-4 |
| rotation | 0.83709 | 0.83680 | 0.99917 | 2.1e-4 |
| internal | −0.39310 | −0.39314 | 0.99973 | 1.3e-4 |

The one-step mode gain is correct to ~1e-4, the variance to ~1e-3, and the residual is only
~1e-4 of the mode variance. **There is no one-step partition bias and no missing one-step
noise on data states.**

## 3. On-policy pilot (160 autonomous states)

`diagnose_water_bias.py` on 160 states from the 250k Full 100 ps NVT trajectory, with fresh
5 fs SPC/Fw one-step labels (evaluation-only):

| Quantity | Value |
|---|---:|
| input internal KE | 4.484 eV |
| true one-step internal KE | 4.476 eV |
| model one-step internal KE | 4.483 eV |
| internal variance ratio | 1.00145 mean; up to 1.0023 at 20–100 ps |
| signed kinetic error per step | +0.0062 eV mean; +0.014 eV at 20–100 ps |
| velocity RMSE | 1.69e-4 Å/fs (reference states: 1.49e-4) |
| position RMSE | 4.86e-4 Å (reference states: 4.08e-4) |

Even on its own states the model's one-step gain matches the SPC/Fw continuation to ~1e-3.
The residual is a tiny **signed mode-KE surplus** (+0.15% of internal KE per step on average,
~0.3% at 20–100 ps). The reference dynamics also keeps the hot state hot for one step, so this
is not a one-step restoring-force error. CSVR removes the total energy but not the partition;
the surplus accumulates until the global rescaling balances it, leaving the factor-of-2.3
internal excess.

This is the target for a **signed mode-resolved kinetic-energy penalty** on on-policy states
(not a magnitude/MSE penalty, which is what E1 did and it heated the system further).

## 4. Interpretation

The NVT failure is a **closed-loop distribution shift with positive feedback**: a tiny
state-dependent error moves the model off the training distribution; there the error is larger
and biased in the same direction, so the partition runs away until the thermostat balances it.

Consequences:

- endpoint, 2-step, 20 ps NVE, 500 fs rollout, and one-step gain/spectral losses cannot see it;
- the root-cause training signal is on-policy: states the model actually visits in a long
  rollout, labelled with reference one-step continuations (DAgger / Dynamic Training);
- a mode-resolved thermostat can stabilize the symptom but does not fix the state distribution.

See `../LONG_TIME_TRAINING_LITERATURE.md` and `../../research_map/GAP_LOG.md` G14.
