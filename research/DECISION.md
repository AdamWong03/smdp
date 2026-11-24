# Decision record

## 2026-11-23 — decide (Candidate 1)

**Question being decided:** Is a constant mass-weighted covariance-matching readout sufficient to fix the KE variance while preserving ACF and Green–Kubo transport?

**Options considered:**

1. Covariance-matching readout (constant linear transform).
2. Fixed basis-induced rule with stochastic group rescale.
3. Self-consistent GLE / temporal kernel with FD noise.

**Evidence:**

- `covmatch-offline-20261123`; output `experiments/covariance_match_20261123_v2.json`.
- Static covariance matches to ~1e-11, but translation KE std is 0.105 vs reference 0.240; rotation 0.106 vs 0.238; internal 0.186 vs 0.249.
- Translation ACF is not improved; GK is 1.445 vs model 1.420 (reference 100 ps 1.502; full reference 1.88).
- KE variance is a temporal/dynamical quantity, not fixed by static covariance.

**Classification:** unsupported.

**Decision:** reject Candidate 1. Do not implement the covariance-matching readout. Continue the search with a temporal-memory/noise candidate.

**Reason:** matching the static covariance is calibration on a stationary moment; it does not restore the temporal fluctuations or the transport integral.

**Next experiment and budget:**

- First test the backbone acceleration interface for the self-consistent GLE: can finite-difference acceleration from the backbone approximate the instantaneous acceleration/force on held-out NVE data?
- In parallel, test whether a linear PSD kernel + FD noise can match the reference velocity ACF, KE variance, and D in a velocity-only model.
- Budget: CPU/GPU analysis; no large rollout until the interface/feasibility test passes.

**Target change:** none.

## 2026-11-23 — decide (GLE interface)

**Question being decided:** Can a 5 fs displacement define a reliable instantaneous acceleration for a self-consistent GLE?

**Options considered:**

1. Position-consistent GLE with \(a_{\rm backbone}\) from finite differences.
2. Velocity-only GLE readout with FD noise, keeping the backbone displacement.
3. Slow-mode + stochastic group rescale as a cheaper temporal control.

**Evidence:**

- `gle-interface-20261123`: \(a_{\rm fd}\) vs \(a_t\) correlation 0.54, RMSE/std 0.87; \(\Delta v\) vs impulse correlation 0.036.
- The 5 fs finite difference is an interval average, not an instantaneous acceleration; the force changes significantly within the interval.

**Classification:** unsupported / high-risk for the position-consistent GLE at 5 fs.

**Decision:** drop the 5 fs position-consistent GLE. Do not implement it. Continue with the velocity-only GLE readout, which does not require the acceleration interface, and the slow-mode + stochastic group-rescale control.

**Next experiment and budget:**

- Fit a PSD velocity kernel + FD noise to the reference translation ACF; simulate the velocity-only GLE and check KE variance, ACF, and D. CPU/GPU analysis; no large rollout.
- In parallel, run 100 ps slow-mode + stochastic group rescale as a control.
- Only if the velocity-only GLE passes the KE/ACF/D gate, integrate it with the frozen backbone and test the hard cross-material rule.

**Target change:** none.

## 2026-11-23 — decide (slow-mode + stochastic control)

**Question being decided:** Does slow-mode + stochastic group rescale satisfy the water frontier gate?

**Options considered:**

1. Promote the slow + stochastic combination as the water candidate.
2. Reject it as a near-miss and continue with the velocity-only GLE.
3. Tune the slow-mode weight/tau until the gate passes.

**Evidence:**

- `slow-stoch-100ps-20261123`: KE std 0.4405 vs 0.4417; VDOS 0.858/0.970; D 1.480 (21.4% low); GK 1.386 vs MSD 1.480; O–H 0.1253.
- D is just outside the 20% gate; O–H is above the 0.118 frontier baseline.
- The slow mode is water-specific and has no quartz counterpart, so the configuration is not a single rule.

**Classification:** near-miss / unsupported for the hard gate.

**Decision:** do not promote. Do not tune the slow-mode parameters post hoc to pass the gate; that would be per-system tuning. Continue with the velocity-only GLE readout.

**Next experiment and budget:**

- Fit a PSD velocity kernel + FD noise to the reference velocity ACF; standalone check KE variance, ACF, and D.
- If it passes, integrate with the frozen backbone and test the same rule on water and quartz.
- Budget: CPU/GPU analysis; no large rollout until the standalone check passes.

**Target change:** none.

## 2026-11-23 — decide (3-mode velocity GLE fit)

**Question being decided:** Does a 3-mode damped-oscillator + slow-mode GLE fit the reference translation ACF with the correct D?

**Options considered:**

1. Implement the 3-mode GLE sampler.
2. Increase the number of modes.
3. Drop the continuous 3-mode fit and test the AR(10)+slow process without post-hoc rescale.

**Evidence:**

- `velocity-gle-fit-20261123`: integral 11.29 vs 11.34 fs; ACF RMSE 0.0129; 200 fs −0.041 vs −0.080; 500 fs +0.006 vs −0.022.
- The fit misses the negative dip and the long tail; the slow mode collapsed to τ=39.7 fs.

**Classification:** unsupported / limited.

**Decision:** do not implement the 3-mode GLE. Test the AR(10)+slow process with no post-hoc rescale; the AR process already carries FD-consistent noise and may have the correct stationary variance.

**Next experiment and budget:** `glr_slow_norescale_100ps_20261123`; 100 ps, existing code; check T, KE std, RDF/VDOS, D, GK.

**Target change:** none.

## 2026-11-23 — decide (no-rescale control)

**Question being decided:** Can AR(10)+slow, without post-hoc rescale, hold the composed-system temperature?

**Options considered:**

1. Continue the no-rescale run.
2. Add explicit composed-system variance control (global CSVR) after the deterministic mode/group rescale.
3. Increase the readout blend/coupling and retest structure.

**Evidence:**

- `slow-norescale-100ps-20261123`: T=70.1 K at 1k steps; the 0.9 backbone weight dominates and cools.
- The local FD-consistent AR process does not set the composed system's stationary variance.

**Classification:** unsupported.

**Decision:** stop the no-rescale path. The readout needs an explicit composed-system variance control; keep the mode-resolved spectral correction for the partition/VDOS. Test a two-level scheme: deterministic mode/group mean correction + global CSVR on the total KE.

**Next experiment and budget:** `glr_slow_global_100ps_20261123`; 100 ps; add a global Gamma target for total KE after the deterministic group rescale. Check T, total KE std, RDF/VDOS, D, GK.

**Target change:** none.

## 2026-11-23 — decide (global CSVR)

**Question being decided:** Does deterministic group mean correction + global CSVR satisfy the water frontier gate?

**Options considered:**

1. Promote the global CSVR scheme.
2. Add a mode-resolved stochastic component on top of global CSVR.
3. Use a weak-coupling stochastic group rescale with τ≈100 fs.

**Evidence:**

- `slow-global-100ps-20261123`: total KE std 0.4344; mode KE std 0.1455/0.1548/0.1480 vs ref 0.2391/0.2342/0.2499; D 1.246; GK 1.650 (32% gap); O–H 0.105; VDOS 0.855/0.973.
- Global CSVR fixes the total variance but not the mode-resolved variance; it also decouples GK from MSD.

**Classification:** unsupported.

**Decision:** do not promote. Test a weak-coupling stochastic group rescale with τ≈100 fs, matching the KE correlation time; this is the principled middle between deterministic mean correction and fresh per-step Gamma noise.

**Next experiment and budget:** `glr_slow_weak_100ps_20261123`; 100 ps; stochastic group rescale with `--fd-rescale-tau 100`, slow mode on. Check mode KE std, D/GK, RDF/VDOS.

**Target change:** none.

## 2026-11-23 — decide (weak-coupling τ=100, invalid run)

**Question being decided:** Does weak-coupling stochastic group rescale τ=100 satisfy the water frontier gate?

**Evidence:**

- `slow-weak-100ps-20261123`: total KE std 0.0708; mode KE std 0.0397/0.0616/0.0600; D 1.384; GK 1.661.
- The old `rescale_tau` implementation used an exponential moving average of independent Gamma samples. Its stationary variance is `(1-alpha)/(1+alpha)` of the Gamma variance (~2.6% at τ=100); it cannot preserve the target marginal.

**Classification:** implementation error, not a scientific negative.

**Decision:** mark the run invalid. Replace the EMA with a Gamma AR(1):

\[
K_{t+1}=\alpha K_t+\Gamma(a(1-\alpha), b),
\]

which preserves the Gamma(a,b) stationary marginal. Rerun the same τ=100 test.

**Next experiment:** corrected `slow-weak-100ps` rerun; check total and mode KE std, D/GK, RDF/VDOS.

**Target change:** none.

## 2026-11-23 — decide (corrected weak-coupling τ=100)

**Question being decided:** Does corrected Gamma AR(1) weak-coupling group rescale τ=100 satisfy the water frontier gate?

**Evidence:**

- `slow-weak-ar1-100ps-20261123`: total KE std 0.318 vs 0.442; mode KE std 0.187/0.191/0.190 vs 0.239/0.234/0.250; D 1.361; GK 1.530 (12.4% gap); O–H 0.1229; VDOS 0.857/0.973.
- The isolated Gamma AR(1) preserves the target marginal, but the composed system does not realize the full variance.

**Classification:** unsupported for the hard gate.

**Decision:** stop tuning the rescale family. The tension between mode variance, D/GK, and RDF is structural, not a missing τ or weight. Do not run more τ/weight sweeps.

**Next experiment and budget:**

- Test the finite-difference acceleration interface at 1 fs on the 0.5 fs NVE data. If \(a_{\rm fd}\) correlates well with \(F/m\) at 1 fs, a position-consistent GLE at 1 fs is plausible and a 1 fs backbone is worth training. If not, drop that path.
- Budget: CPU analysis first; a 1 fs backbone only after the interface passes.

**Target change:** none.

## 2026-11-23 — decide (1 fs vs 2 fs vs 5 fs interface)

**Question being decided:** At what horizon can a position-consistent GLE use the backbone displacement as an instantaneous drift?

**Options considered:**

1. 1 fs backbone + position-consistent GLE.
2. 2 fs backbone as a speed/accuracy compromise.
3. Keep the 5 fs decoupled readout and accept that the hard gate is not met.

**Evidence:**

- `interface-resolution-20261123`: 1 fs corr 0.987 (a) / 0.948 (Δv); 2 fs 0.920/0.798; 5 fs 0.541/0.036.
- The 5 fs position-consistent GLE was already rejected by `gle-interface-20261123`.

**Classification:** 1 fs is a credible interface; 2 fs is marginal; 5 fs is not.

**Decision:** the 5 fs position-consistent GLE is rejected. The 1 fs path is a Gate B architecture/budget decision: it requires a new 1 fs backbone (water and quartz) and a position-consistent GLE integrator. Do not start training without human approval.

**Trade-off:** a 1 fs horizon reduces the speed advantage (roughly 2× vs 0.5 fs MD instead of ~10× at 5 fs), but it is the only horizon where the GLE drift interface is defensible.

**Next action:** ask the human to choose 1 fs, 2 fs, or stop the position-consistent path.

**Target change:** none yet; the objective remains credible long-time MD.

## 2026-11-23 — decide (Gate B: 1 fs architecture)

**Question being decided:** Which horizon should the position-consistent GLE use?

**Options considered:**

1. 1 fs backbone + position-consistent GLE: interface corr 0.987, speedup ~2×.
2. 2 fs backbone: interface corr 0.920, speedup ~4×, higher risk.
3. Stop the position-consistent path; keep the 5 fs decoupled readout as a negative/contrast result.

**Evidence:**

- `interface-resolution-20261123`: 1 fs corr 0.987/0.948; 2 fs 0.920/0.798; 5 fs 0.541/0.036.
- The 5 fs decoupled readout family is exhausted; no configuration passes the water frontier gate.

**Decision:** choose 1 fs. Train a 1 fs water backbone, then implement a position-consistent GLE. Do not start GLE before the backbone passes its test metrics.

**Budget:** 1–2 days GPU for the water backbone; quartz only after the water GLE gate passes. The speed advantage is ~2× vs 0.5 fs MD, not ~10× at 5 fs.

**Next action:** `water-1fs-backbone-20261123` is running; after it finishes, test the backbone and build the 1 fs GLE kernel/integrator.

**Target change:** none.

## 2026-11-24 — decide (1 fs integrator design)

**Question being decided:** Which 1 fs position-consistent integrator is stable enough to test?

**Options considered:**

1. Blend the backbone and AR(50) velocity, integrate positions with the blend.
2. Driven AR-GLE: \(v'=v_{\rm AR}+h a_b\).
3. Langevin drift: \(v'=v_{\rm pred}-\gamma h v+\sqrt{2\gamma h k_BT/m}\xi\).
4. Langevin drift + global CSVR.

**Evidence:**

- Blend: T=70.9 K, KE std 1.10 (negative; 0.9 backbone weight dominates).
- Driven AR-GLE: unstable at 4 steps, preT 3509 K (negative; AR predictor is not a passive memory kernel).
- Langevin \(\gamma=0.1\): stable but T=256.8 K, KE std 0.395 (cold).
- Langevin + global CSVR: 1 ps smoke T=300.61 K, KE std 0.4354.

**Classification:** options 1 and 2 are negative; option 3 is stable but biased; option 4 is the current candidate.

**Decision:** use Langevin drift + global CSVR for the 100 ps water gate. If it passes, add a PSD memory kernel for the multi-lag ACF. If it fails, reconsider the position-consistent architecture.

**Next experiment:** `water-1fs-langevin-csVR-100ps-20261124`; analyze T/mode KE/RDF/VDOS/D/GK.

**Target change:** none.

## 2026-11-24 — decide (1 fs Langevin + global CSVR)

**Question being decided:** Does Langevin drift + global CSVR satisfy the water frontier gate at 1 fs?

**Evidence:**

- `water-1fs-langevin-csVR-100ps-20261124`: T=301.47; partition 3.04/2.90/1.50 vs 2.37/2.43/2.60; RDF O–H 0.862; VDOS 0.520/0.442; D 4.13e-10; GK 4.72e-10.
- Global CSVR fixes the total KE but the Langevin drift destroys the mode partition and structure.

**Classification:** unsupported.

**Decision:** stop the Langevin+CSVR path. The last bounded candidate is a proper PSD GLE kernel at 1 fs, fitted to the reference ACF/D and integrated with the backbone drift. If the 1 fs PSD fit fails, do not add more integrator variants; consider end-to-end training with the integrator in the loop or a target change.

**Next experiment:** fit the 1 fs reference ACF with 4–6 damped oscillators + slow mode, constrained to the integral; check the ACF at 5/25/100/200/500/1000/5000 fs. CPU only.

**Target change:** none.

## 2026-11-24 — decide (PSD kernel fit at 1 fs)

**Question being decided:** Is a post-hoc PSD GLE kernel viable at 1 fs?

**Evidence:**

- `velocity-gle-fit-1fs-20261124`: 4 modes ACF@5 fs 0.315 vs 0.981, RMSE 0.071; 5 modes 0.346 vs 0.981, RMSE 0.066.
- The short-time ACF is not fit; the representation cannot simultaneously capture the short decay and the long negative dip.

**Classification:** unsupported.

**Decision:** stop adding post-hoc integrator/kernel variants. The frozen-backbone + readout/integrator family has failed the hard gate at both 5 fs and 1 fs. The remaining principled path is **end-to-end training with the integrator in the loop**, so the drift learns to be consistent with the thermostat/stationary distribution. A target change to a diagnostic paper is the alternative but was explicitly rejected.

**Next action:** ask the human for a Gate B decision on end-to-end training vs stopping/pivoting. Do not start a new training project without approval.

**Target change:** none yet.
