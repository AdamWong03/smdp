# Long-Time / Equilibrium Training of Learned MD Propagators — Literature Synthesis

**Date:** 2026-11-21
**Status:** literature scan complete; method decision pending
**Scope:** how related work handles long-horizon stability, equilibrium statistics, and
closed-loop distribution shift in learned MD / learned simulators.

> **Correction (2026-11-21, after the user's challenge and the published-checkpoint audit):**
> the earlier recommendation to move to on-policy correction was premature. TrajCast is
> **offline** and reports faithful water equilibrium; on-policy is one of several ways to
> reduce the state-dependent error, not a necessary condition. The decisive variable is the
> **mode-resolved one-step error relative to the closed-loop amplification**. See
> `PAPER_PLAN_20261121.md` and the addendum at the end of this document.

---

## 0. What has to be explained (updated diagnostic)

Three new diagnostics change the picture. All are analyses of existing artifacts; no training.

### 0.1 The failure is a mode-partition failure, not a total-energy failure

100 ps NVT production (50 ps thermalization), mode kinetic energy (eV, 64 molecules;
equipartition 2.48 eV per mode) and mass-weighted velocity autocorrelation at lag τ:

| Model | trans KE | rot KE | int KE | total | α_t(25 fs) | α_r(25 fs) | α_i(5 fs) | α_i(100 fs) | α_i(500 fs) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference | 2.373 | 2.430 | 2.609 | 7.411 | 0.658 | −0.540 | −0.567 | 0.147 | 0.104 |
| 250k Full | 0.538 | 0.786 | 6.044 | 7.368 | 0.393 | −0.783 | −0.902 | 0.831 | 0.722 |
| Current-Only | 0.646 | 0.535 | 6.384 | 7.565 | 0.646 | −0.764 | −0.917 | 0.858 | 0.590 |
| Native ep145 | 3.256 | 3.105 | 1.041 | 7.402 | 0.697 | −0.412 | +0.109 | −0.288 | 0.003 |

Equipartition at 300 K is ≈2.48 eV per mode. Every model has approximately the correct
**total** kinetic energy (CSVR enforces it) but the wrong **partition**:

- our models: collective modes cold, internal modes hot and **underdamped**
  (internal ACF still 0.72 at 500 fs — the internal oscillators barely dephase);
- native ep145: collective modes hot, internal modes cold and **overdamped**.

This is not a temperature or energy-scale bug. It is a mode-resolved energy imbalance that the
closed-loop map (propagator + CSVR) amplifies.

### 0.2 The one-step map on held-out data is essentially perfect

Held-out NVE pairs, 250k Full, mass-weighted mode statistics:

| Mode | α_ref | α_pred | Var ratio | residual / reference variance |
|---|---:|---:|---:|---:|
| translation | 0.98133 | 0.98154 | 1.00044 | 2.1e-4 |
| rotation | 0.83709 | 0.83680 | 0.99917 | 2.1e-4 |
| internal | −0.39310 | −0.39314 | 0.99973 | 1.3e-4 |

The one-step gain is correct to ~1e-4, the mode variance to ~1e-3, and the model residual is
~1e-4 of the mode variance (RMS error ~1–2% of the velocity RMS). **There is no missing
one-step noise and no one-step partition bias on data states.**

The contrast with the rollout is the key: the same model has internal α(5 fs) = −0.90 on its
own NVT trajectory (table above) but −0.39 on held-out NVE pairs (table here). The map is
accurate where it was trained and qualitatively different where it actually runs. (The NVE
one-step α is not identical to the NVT ACF even for the reference, because CSVR rescaling
changes the short-time correlation; the relevant comparison is model-on-data vs
model-on-own-states.)

### 0.3 On-policy pilot: locally accurate, globally biased

`diagnose_water_bias.py` was run on 160 states sampled from the 250k Full 100 ps NVT
trajectory (32 states in each of five windows), with fresh 5 fs SPC/Fw one-step labels
(evaluation-only pilot).

| Quantity (mean over 160 states) | Value |
|---|---:|
| input internal KE | 4.484 eV |
| true one-step internal KE | 4.476 eV |
| model one-step internal KE | 4.483 eV |
| internal variance ratio (model/true) | 1.00145 mean; up to 1.0023 at 20–100 ps |
| signed kinetic error per step | +0.0062 eV mean; +0.014 eV at 20–100 ps |
| velocity RMSE | 1.69e-4 Å/fs (reference states: 1.49e-4) |
| position RMSE | 4.86e-4 Å (reference states: 4.08e-4) |

Even on its own states the model's one-step **gain** matches the SPC/Fw continuation to ~1e-3.
The residual is a tiny **signed mode-KE surplus**: +0.15% of the internal KE per step on
average, rising to ~0.3% in the 20–100 ps windows. The reference dynamics also keeps the hot
state hot for one step, so this is not a one-step restoring-force error; it is a slow bias.
CSVR removes the total energy but not the partition, so the surplus accumulates until the
global rescaling balances it, leaving a factor-of-2.3 internal excess.

This is the precise target for a training fix: a **signed mode-resolved kinetic-energy
penalty** on on-policy states, not a magnitude/MSE penalty (which is what E1 did, and it
heated the system further).

### 0.4 Consequence

The failure is a **closed-loop distribution shift with positive feedback**:

1. a tiny state-dependent one-step error shifts the mode partition slightly;
2. the shifted state is off the training distribution;
3. there the error is larger and biased in the same direction;
4. the partition runs away until the thermostat balances it.

Short-horizon losses cannot see this (one-step and 2-step partitions are already correct), and
neither can a one-step gain/spectral regularizer (the one-step gain is already correct).
The training signal must come from **states the model actually visits in a long rollout**.

---

## 1. Strategy families in related work

### A. On-policy / learner-induced data (DAgger, Dynamic Training, active learning)

**Idea:** train on the state distribution induced by the learner, not only on reference
trajectories.

- **DAgger** (Ross, Gordon & Bagnell, AISTATS 2011, arXiv:1011.0686): iteratively run the
  learner, label visited states with the expert, aggregate the dataset, retrain. No-regret
  bound on the learner's cost under the induced distribution.
- **Dynamic Training** (Žugec et al., *J. Chem. Inf. Model.* 65, 8033, 2025,
  arXiv:2504.03521): the MD analogue. Run MD with the current MLIP, recompute reference
  (DFT) labels for visited configurations, retrain. Architecture-independent; improves
  accuracy over extended MD.
- **Uncertainty-driven active learning** (Kulichenko et al., *Nat. Commun.* 2023): use model
  uncertainty to bias dynamics toward underrepresented regions. Useful for data selection,
  not by itself for equilibrium statistics.

**Guarantee:** reduces error on the learner's own state distribution; DAgger has a no-regret
bound. It does **not** guarantee the invariant measure unless the loop is iterated to a fixed
point where the model is accurate on its own stationary distribution.

**Fit to our failure:** highest. The one-step map is correct on data but not on its own states;
this is exactly the DAgger setting. The fixed point we want is "model accurate on the NVT
states it generates."

**Cost:** one 100 ps NVT rollout (20 min) + reference one-step labels on sampled states
+ fine-tune (hours). Reference engine already exists (`diagnose_water_bias.py`,
`physics/spcfw.py`).

**Rule conflict:** the project currently labels these reference continuations
"evaluation only; never training." Using them for training needs explicit approval.

### B. Exposure bias / scheduled sampling / multi-step (pushforward) training

**Idea:** during training, feed the model its own predictions, or train on multi-step rollouts.

- **Scheduled sampling** (Bengio et al., NeurIPS 2015, arXiv:1506.03099): gradually replace
  ground-truth inputs with model predictions.
- **Pushforward trick / temporal bundling** (Brandstetter, Worrall & Welling, ICLR 2022,
  arXiv:2202.03376): train on model-generated states to encourage zero-stability of
  autoregressive PDE solvers.
- **Graph network simulators** (Sanchez-Gonzalez et al., ICML 2020): noise injection during
  training improves long-rollout stability.
- **Caveat:** scheduled sampling is known to be an inconsistent objective (Huszár, 2015,
  arXiv:1511.05101); it helps exposure bias but does not target a stationary distribution.

**Fit:** relevant mechanism, but **insufficient alone** in our setting. Our 2-step rollout
with a partition loss was null because the 2-step partition is already correct. Pushforward
horizons of 2–10 steps are far shorter than the ~20,000-step feedback loop that produces the
bias. A pushforward version would need to reach the relaxation time of the slow modes
(at least tens of ps), which is expensive and unstable with backprop through time.

**Cost:** low for short horizons (already tried, null); high for long horizons.

### C. Stochastic / probabilistic propagators and generative trajectory models

**Idea:** learn the conditional **distribution** p(x_{t+Δ}|x_t), not only its mean.

- **Timewarp** (Klein et al., NeurIPS 2023, arXiv:2302.01170): a normalizing-flow proposal
  used inside MCMC targeting the Boltzmann distribution. Correct equilibrium is guaranteed by
  the MCMC correction, not by the network; the flow only needs to be a good proposal.
- **ITO** (Schreiner, Winther & Olsson, NeurIPS 2023, arXiv:2305.18046): stochastic
  diffusion-model surrogates learned at multiple time resolutions; self-consistent across
  time scales.
- **MDGen** (Jing et al., NeurIPS 2024): generative modeling of MD trajectories; samples
  trajectories rather than deterministic rollouts.
- **Boltzmann generators** (Noé et al., *Science* 365, eaaw1147, 2019): one-shot equilibrium
  sampling with exact likelihood/reweighting; no dynamics.
- **LiFlow** (Nam et al., *Nat. Mach. Intell.* 7, 1625, 2025): flow matching for accelerated
  atomistic simulation.

**Guarantee:** a correctly learned stochastic kernel can have the right invariant measure;
Timewarp's MCMC wrapper guarantees it by construction (given an energy function).

**Fit:** conceptually right, but our one-step residual is only ~1e-4 of the mode variance, so
the missing one-step stochasticity is not the dominant error. Adding noise without fixing the
state distribution would not address the feedback loop. This family becomes relevant if
on-policy correction stalls.

**Cost:** medium–high (new architecture/objective); MCMC correction needs an energy model,
which breaks the force-free setting.

### D. Structure-preserving priors and physical thermostats

**Idea:** constrain the map or the integrator so conservation/equilibrium holds by construction.

- **Hamiltonian neural networks** (Greydanus et al., NeurIPS 2019) and **symplectic recurrent
  networks** (Chen et al., ICLR 2020, arXiv:1909.13334): preserve energy/phase-space volume
  in the conservative limit.
- **Symplectic integrators + Langevin/CSVR thermostats:** the canonical measure is preserved
  if the deterministic drift is conservative and the noise satisfies fluctuation–dissipation.
- **Mode-resolved thermostats** are standard in MD for flexible water, where a single global
  thermostat can produce mode-specific equipartition errors.
- **TrajCast** (Thiemann et al., *Nat. Mach. Intell.* 8, 764, 2026,
  DOI 10.1038/s42256-026-01227-7): trains on NVE, applies CSVR after every step; enforces
  linear momentum conservation; reports water VDOS overlap ≈0.99 and diffusion 2.18 vs
  1.84 × 10⁻⁹ m² s⁻¹ over 2.5 ns with 5,000 training samples at Δt = 5 fs.

**Guarantee:** structure-preserving models cannot drift in the conserved quantities, but a
learned Hamiltonian is an MLIP (forces), which is the setting we want to avoid. A
mode-resolved thermostat can enforce equipartition but may distort dynamics.

**Fit:** a mode-resolved thermostat is a plausible **stabilizer**, not a root-cause fix. It
would cap the runaway but would not make the model accurate on its own states.

**Cost:** low for a thermostat layer; medium for a Hamiltonian/symplectic reparameterization.

### E. Spectral / transfer-operator consistency

**Idea:** constrain long-time statistics (autocorrelations, slow modes, invariant measure)
directly, instead of only one-step errors.

- **VAMPnets** (Mardt et al., *Nat. Commun.* 9, 5, 2018): variational lower bound on the
  transfer operator; learns slow collective variables and their spectrum.
- **ITO** (above): consistency across multiple lag times.
- **Koopman/transfer-operator autoencoders:** fit the propagator's action on observables.

**Guarantee:** matching the transfer operator's spectrum at several lags constrains the
relaxation to equilibrium; the stationary distribution is the top eigenfunction.

**Fit:** attractive in principle, but our one-step gain is already correct to 1e-4. A
spectral loss at short lags would see nothing; a loss at long lags needs long model rollouts,
which brings back the same cost/stability problem as on-policy training. It could be a useful
**regularizer inside** an on-policy loop, not a replacement.

**Cost:** medium; data-efficient (reuses existing trajectories).

### F. Equilibrium by construction (MCMC correction, reweighting)

**Idea:** never trust the propagator for equilibrium; correct it.

- **Timewarp + Metropolis–Hastings:** accept/reject with the Boltzmann target.
- **Boltzmann generators + importance reweighting:** correct a flow's bias with likelihood
  ratios.
- **Umbrella sampling / replica exchange:** enhance sampling around the learned dynamics.

**Guarantee:** asymptotically exact equilibrium, independent of the propagator's bias.

**Fit:** the strongest guarantee, but requires an energy function (SPC/Fw or a learned
potential) at every step, which destroys the force-free speed advantage. Could be used as an
offline validation/repair tool, not as the main method.

**Cost:** high at inference.

---

## 2. The decisive external data point: published TrajCast succeeds on water

The TrajCast paper (NMI 2026) reports, for the **same system class** (64 flexible SPC/Fw
water molecules, 300 K, 5 fs horizon, CSVR τ = 50 fs):

- 5,000 training configurations from three independent 50 ps NVE trajectories;
- 1,500 epochs max, batch 2, Adam lr 0.01, double precision, up to 11 days on a V100;
- 2.5 ns NVT rollout (500,000 steps);
- VDOS overlap ≈0.99 for H and O;
- diffusion 2.18 ± 0.19 vs 1.84 ± 0.03 × 10⁻⁹ m² s⁻¹;
- RDF agreement across all pairs, slight O–O understructuring;
- MSD diverges slightly at long lag, attributed to error accumulation.

**Implication:** a direct, force-free, 5 fs propagator **can** reproduce water's equilibrium
partition. Our failure is not an intrinsic impossibility of the approach. Our native control
is at epoch 145/1500 (372,500 of ~3.75M optimizer steps, ~10% of the schedule), with ~1%
one-step velocity error. It is **not** the published model, and its opposite bias cannot be
used as evidence that the architecture fails.

Two consequences for the paper:

1. The baseline must eventually be a **converged native TrajCast**, not ep145. Reviewers will
   reject a comparison against a 10%-trained baseline.
2. The method contribution cannot be "direct propagators work"; that is the published result.
   It must be "how to reach faithful long-time equilibrium **reliably and cheaply**, and why
   the naive one-step training fails." The closed-loop diagnostic is the novel part.

**Practical note:** the native run is slow because it uses float64 on an L40S (weak FP64).
A float32 resume with the same recipe should be ~5–8× faster; early stopping may trigger well
before 1,500 epochs. This is a control run, not the method.

---

## 3. Mapping strategies to our failure

| Strategy | Targets our mechanism? | Evidence / precedent | Cost | Decision |
|---|---|---|---|---|
| **A. On-policy correction (DAgger / Dynamic Training)** | **Yes** — directly trains on the visited, off-distribution states; breaks the feedback loop | DAgger; Žugec 2025; our one-step-on-data vs rollout gap | 100 ps rollout + labels + fine-tune (hours) | **Recommended**, needs approval |
| B. Scheduled sampling / pushforward (short) | Partly — exposure bias, but horizon too short | Bengio 2015; Brandstetter 2022; our 2-step null | low | Not sufficient alone |
| B'. Long pushforward / TBPTT through ps | Yes in principle | unstable, expensive | very high | Fallback |
| C. Stochastic propagator | Only if one-step noise is missing (it is not) | Timewarp, ITO, MDGen | medium–high | Not first |
| D. Mode-resolved thermostat / FD layer | Stabilizes the symptom, not the state distribution | MD thermostat practice; TrajCast CSVR | low | Possible stabilizer inside A |
| E. Spectral / transfer-operator loss | Not at short lags (gain already correct); long lags need rollouts | VAMPnets, ITO | medium | Regularizer inside A |
| F. MCMC / reweighting | Exact equilibrium, ignores dynamics | Timewarp, Boltzmann generators | high at inference | Validation tool only |

---

## 4. Recommended method: on-policy signed mode-KE calibration

The literature and the new diagnostic point to a specific design:

**Closed-loop, mode-targeted correction of the signed kinetic-energy partition.**

1. Run a 100 ps NVT rollout with the current model (already implemented).
2. Sample states from the rollout, **oversampling the drifting windows** (internal-hot /
   collective-cold) rather than uniformly.
3. At each sampled state, compute a one-step reference continuation with SPC/Fw
   (`diagnose_water_bias.py` already does exactly this).
4. Fine-tune on a mixture of reference-trajectory pairs and on-policy pairs with a
   **signed mode-resolved kinetic-energy loss**:

   ```
   L_KE = Σ_mode [ (KE_pred,mode - KE_true,mode) / KE_scale ]²
   ```

   Unlike a magnitude/MSE loss (E1), this penalizes the *direction* of the per-step bias.
   Add the ordinary endpoint loss and a trust-region anchor to the base model; keep a replay
   buffer of reference pairs to prevent forgetting.
5. Re-run the 100 ps NVT screen; iterate 2–3 rounds until the partition is stationary and
   matches the reference.
6. Validate at 2.5 ns: partition, RDF, VDOS, diffusion.

**Cheap control (no approval needed): mode-resolved thermostat.**

Before or alongside the fine-tune, add a mode-resolved canonical rescaling after each
propagator step (translation / rotation / internal each relaxed to 1/2 kT per DOF). This
enforces the partition at every step and directly tests whether the partition is the dominant
cause of the RDF/VDOS/diffusion failure. It is a patch, not the method, but it is one 100 ps
screen (20 min) and it de-risks the training run.

**Why this is a method paper, not a data patch:**

- The diagnostic identifies a **closed-loop instability** that one-step, two-step, endpoint,
  and spectral objectives cannot see. This is a general property of learned propagators, not
  a water-specific bug.
- The correction is **mode-targeted and signed**: it samples the states where the invariant
  measure is drifting and penalizes the direction of the per-step KE bias, which is more
  sample-efficient than uniform DAgger or magnitude MSE.
- The result is a **force-free** model at inference: reference labels are used only during
  training, exactly like offline expert labels in imitation learning.
- Ablations: uniform vs mode-targeted sampling; signed KE vs magnitude MSE; 1 vs 2 vs 3
  rounds; replay vs no replay; on-policy only vs mixed; comparison with a mode-resolved
  thermostat.

**Risks:**

- Reference one-step labels at off-distribution states may be expensive or noisy (SPC/Fw is
  cheap here, so this is manageable).
- The fine-tune may drift on reference states (catastrophic forgetting); replay and a
  trust-region anchor are needed.
- If the fixed point does not exist at the current model capacity, on-policy correction will
  plateau; then the fallback is a capacity/backbone change.

**Decision rule:**

- Round 1 moves the 100 ps partition toward reference and keeps RDF/VDOS/diffusion stable →
  continue to 2.5 ns and make it the method.
- Partition moves but dynamics degrade → add the mode-resolved thermostat/stabilizer.
- No movement or instability → architecture/capacity limit; stop on-policy and change the
  backbone.


---

## 5. What not to do

- Do not tune endpoint, 2-step, 20 ps NVE, or 500 fs rollout losses further; all are blind to
  the closed-loop drift (two independent nulls).
- Do not claim the native architecture fails on the basis of ep145; it is 10% trained.
- Do not add a one-step gain/spectral regularizer as the main fix; the one-step gain is
  already correct.
- Do not use a global thermostat tweak as the method; a mode-resolved thermostat is only a
  control. It can mask the failure without fixing the state distribution.

---

## 6. References

1. Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to
   No-Regret Online Learning. AISTATS 2011. arXiv:1011.0686.
2. Bengio, Vinyals, Jaitly, Shazeer. Scheduled Sampling for Sequence Prediction with
   Recurrent Neural Networks. NeurIPS 2015. arXiv:1506.03099.
3. Huszár. How (not) to train your generative model: scheduled sampling, likelihood,
   adversary? 2015. arXiv:1511.05101.
4. Žugec, Hadži Veljković, Alducin, Juaristi. Dynamic Training Enhances Machine Learning
   Potentials for Long-Lasting Molecular Dynamics. J. Chem. Inf. Model. 65, 8033 (2025).
   arXiv:2504.03521.
5. Brandstetter, Worrall, Welling. Message Passing Neural PDE Solvers. ICLR 2022.
   arXiv:2202.03376.
6. Sanchez-Gonzalez et al. Learning to Simulate Complex Physics with Graph Networks.
   ICML 2020.
7. Thiemann, Reschützegger, Esposito, Taddese, Olarte-Plata, Martelli. Force-free molecular
   dynamics through autoregressive equivariant networks. Nat. Mach. Intell. 8, 764–776
   (2026). DOI:10.1038/s42256-026-01227-7.
8. Klein et al. Timewarp: Transferable Acceleration of Molecular Dynamics by Learning
   Time-Coarsened Dynamics. NeurIPS 2023. arXiv:2302.01170.
9. Schreiner, Winther, Olsson. Implicit Transfer Operator Learning: Multiple Time-Resolution
   Surrogates for Molecular Dynamics. NeurIPS 2023. arXiv:2305.18046.
10. Jing et al. Generative Modeling of Molecular Dynamics Trajectories. NeurIPS 2024.
11. Noé, Olsson, Köhler, Wu. Boltzmann generators: Sampling equilibrium states of many-body
    systems with deep learning. Science 365, eaaw1147 (2019).
12. Mardt, Pasquali, Wu, Noé. VAMPnets for deep learning of molecular kinetics.
    Nat. Commun. 9, 5 (2018).
13. Greydanus, Dzamba, Yosinski. Hamiltonian Neural Networks. NeurIPS 2019.
14. Chen, Shi, Zhang, Ribas, Kilbertus, Büttner. Symplectic Recurrent Neural Networks.
    ICLR 2020. arXiv:1909.13334.
15. Nam et al. Flow matching for accelerated simulation of atomic transport in materials.
    Nat. Mach. Intell. 7, 1625–1635 (2025).
16. Kulichenko et al. Uncertainty-driven dynamics for active learning of interatomic
    potentials. Nat. Commun. 2023.

---

## 7. Addendum — on-policy is not the only answer

The earlier sections over-weighted on-policy correction. The corrected view:

**No, not every method without on-policy data has this problem.** A deterministic propagator
is a closed-loop map when composed with a thermostat. Its stationary mode partition is
controlled by the mode-resolved one-step error relative to the amplification `1/(1-α²)`:

| Mode | α | amplification | sensitivity |
|---|---:|---:|---:|
| translation | 0.981 | 27.0 | 53 |
| rotation | 0.837 | 3.34 | 5.6 |
| internal | −0.393 | 1.18 | 0.93 |

A method avoids the problem if any of the following holds:

1. **Its one-step error is below the threshold** — the published TrajCast model is offline and
   reports 0.37% relative velocity MAE; our native ep145 is 0.50% and fails; our 250k Full is
   ~1.2% and fails much more. This is the simplest explanation and the first one to test.
2. **It is structure-preserving by construction** — Hamiltonian/symplectic dynamics with a
   Langevin thermostat, or a mode-resolved canonical thermostat, has the right invariant
   measure by construction (at the cost of forces or a thermostat layer).
3. **It learns a stochastic kernel** — Timewarp/ITO/MDGen learn conditional distributions;
   MCMC or reweighting can correct the remaining bias (at higher cost).
4. **It trains in the evaluation ensemble** — e.g. thermostatted/equilibrium data, or
   thermostat-in-the-loop training.

On-policy data is a way to reduce the state-dependent error when the threshold cannot be
reached offline. It is not a defining property of correct methods, and it should be the
fallback, not the default.

The current plan is the **accuracy-threshold route**: quantify the threshold, identify the
controlling mode-resolved error, and optimize it offline. See `PAPER_PLAN_20261121.md`.

