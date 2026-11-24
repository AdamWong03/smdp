# Phase 3 — Argument Blueprint

**Date:** 2026-11-22
**Target:** Nature Machine Intelligence, method paper
**Purpose:** claim–evidence–warrant chains, logical flow, counter-arguments, and falsification
conditions. This is the bridge between `OUTLINE.md` / `EVIDENCE_MAP.md` and the Phase 4 draft.

---

## 1. Central thesis

> Force-free direct propagators fail at long times not primarily because they are pointwise
> inaccurate, but because the **composed propagator + thermostat** has a biased stationary
> measure across molecular modes. A **mode-resolved generalized-Langevin readout (GLR)** on a
> frozen propagator can enforce the canonical mode partition and match multi-lag dynamics,
> decoupling what the backbone must learn (structure) from what the readout guarantees
> (thermodynamics).

The thesis has three parts, and each needs its own evidence:

1. **Diagnostic:** pointwise accuracy and short-horizon stability do not predict long-time
   ensemble fidelity.
2. **Mechanism:** the failure is a mode-resolved stationary-measure bias, not an energy-drift or
   one-step-error problem.
3. **Remedy:** a mode-resolved AR(p) FD readout with exact variance rescaling fixes the measure
   without retraining the backbone.

---

## 2. Claim hierarchy

| ID | Claim | Role | Status |
|---|---|---|---|
| C0 | Long-time fidelity is an ensemble property, not a trajectory or one-step property. | framing | supported |
| C1 | Pointwise one-step accuracy is neither necessary nor sufficient for long-time equilibrium. | diagnostic | supported |
| C2 | NVE energy stability is not the differentiator; the composed NVT stationary measure is. | diagnostic | supported |
| C3 | The failure is a mode-resolved partition bias of the composed propagator + thermostat. | mechanism | supported |
| C4 | A mode-resolved AR(p) FD readout guarantees the canonical kinetic partition and matches multi-lag mode ACFs. | method | supported (diffusion pending) |
| C5 | The GLR reproduces water structure and VDOS at or beyond the published checkpoint. | result | supported at 100 ps |
| C6 | The GLR recovers the diffusion coefficient and is initial-condition independent. | result / gate | 100 ps independent; 2.5 ns running |
| C7 | The readout is backbone-agnostic and training-free, with a short reference-NVT data requirement. | scope | partially supported |

---

## 3. Claim–evidence–warrant chains

### C1 — Pointwise accuracy is neither necessary nor sufficient

- **Evidence:** published TrajCast has ~15× larger pointwise error than the 250k model on our
  validation transitions, yet passes the 100 ps NVT screen (partition 2.505/2.463/2.444 eV
  against the 2.5 ns reference 2.459/2.502/2.450 eV); the 250k model has lower pointwise error
  but fails (0.54/0.79/6.04 eV).
- **Warrant:** if one-step MSE were the controlling variable, the pointwise-better model should
  have the better ensemble. The ranking is reversed. Therefore one-step accuracy cannot be
  necessary or sufficient.
- **Qualifier:** demonstrated on one system (SPC/Fw water) and one model pair; the claim is
  existence, not a universal anti-correlation.
- **Falsifier:** a pointwise-better model that also passes, or a pointwise-worse model that fails,
  would not refute the "neither necessary nor sufficient" claim; only a controlled sweep showing
  MSE predicts ensemble error would.
- **Figure:** Fig. 1a,b.

### C2 — NVE stability is a red herring

- **Evidence:** in 25 ps NVE the published model heats 302→476 K and the 250k base cools
  302→270 K; the published model nevertheless passes NVT, the base fails.
- **Warrant:** NVE energy drift and NVT ensemble fidelity are different properties of the
  composed map. A model can be a poor energy integrator and a good canonical sampler, or vice
  versa.
- **Qualifier:** 25 ps NVE is a diagnostic, not a production test; longer NVE may differ.
- **Falsifier:** a model with zero NVE drift that still fails NVT would strengthen the claim; a
  perfect correlation between NVE drift and NVT error across many models would weaken it.
- **Figure:** Fig. 1c,d.

### C3 — The failure is a mode-resolved partition bias

- **Evidence:** v0 is stable for 2.75 ns but has translation 0.111 / rotation 0.808 / internal
  6.338 eV; 250k and Current-Only fail in the same direction; native ep145 fails in the opposite
  direction; M0 overshoots (3.37/3.27/1.00) and is unstable on an independent start; the CSRV
  moment loss blows up at 4.3 ps. The total kinetic energy can be correct while the partition is
  wrong.
- **Warrant:** the failure is not a single scalar (total temperature) but a vector of mode
  temperatures. A global thermostat can hold total T while leaving the mode partition biased.
- **Qualifier:** the mode decomposition is molecular translation/rotation/internal; other
  decompositions (e.g. phonon modes) are not tested.
- **Falsifier:** a model with correct total T and correct mode partition that still fails
  structure would show the partition is not the controlling variable.
- **Figure:** Fig. 1e, Table 1.

### C4 — The GLR guarantees the canonical kinetic partition and matches multi-lag ACFs

- **Evidence:** derivation of the per-mode target \(N_a k_B T/2\) and the Yule–Walker
  prediction-error variance; 100 ps partition 2.482/2.526/2.399 eV (val_04) and
  2.445/2.490/2.362 eV (independent); translation ACF 0.981/0.640/0.014/−0.030 vs reference
  0.982/0.658/0.039/−0.019; rotation and internal short-lag match; p=20 matches internal
  100/200 fs.
- **Warrant:** the rescale sets each mode's instantaneous kinetic energy to its canonical value;
  the AR(p) stationary ACF equals the fitted reference ACF by construction. The empirical
  agreement confirms the implementation.
- **Qualifier:** the guarantee is for the **mean kinetic mode partition** and the **velocity ACF**,
  not the full joint position–velocity distribution. The per-step rescale suppresses the
  equilibrium fluctuations of the mode kinetic energy (reference std ~0.24 eV vs readout
  ~0.002 eV), so the guarantee is on the first moment. The position marginal remains learned.
- **Falsifier:** a rescale-on run with a mode partition off by more than finite-sample error, or
  an ACF that does not follow the AR(p) prediction.
- **Figure:** Fig. 2, Fig. 3b,c.

### C5 — Structure and VDOS at or beyond the published checkpoint

- **Evidence:** 100 ps O–O 0.032, O–H 0.104, H–H 0.026; VDOS 0.848/0.964; independent start
  0.036/0.098/0.027; VDOS 0.851/0.963; published 0.043/0.118/0.096; VDOS 0.624/0.937.
- **Warrant:** lower RDF RMSE and higher VDOS overlap against the same reference mean the
  generated ensemble is closer to the reference. The GLR run is not a different reference or a
  different metric.
- **Qualifier:** H VDOS overlap is sensitive to our reference H-stretch frequency (published
  scores 0.624 here vs ~0.99 in its own analysis); O VDOS and the RDFs are the robust
  comparisons. Single seed.
- **Falsifier:** a seed study showing the GLR RDF/VDOS advantage is within seed noise, or an
  analysis-protocol artifact that reverses the ranking.
- **Figure:** Fig. 3d–g.

### C6 — Diffusion and initial-condition independence (gate)

- **Evidence:** translation ACF now matches at all lags; FDR v1 (AR(1)) gave D=3.59e-9 due to a
  long translation tail; GLR 2.5 ns is running. Independent 100 ps reproduces partition, ACF,
  RDF, VDOS.
- **Warrant:** the Green–Kubo diffusion coefficient is the integral of the velocity ACF. Matching
  the ACF at all lags should bring D close to the reference; if it does not, the residual
  cross-mode correlations or the position dynamics matter.
- **Qualifier:** 2.5 ns is still shorter than the reference; block analysis and multiple seeds
  are needed.
- **Falsifier:** D outside the reference band (e.g. >2.5e-9 or <1.4e-9) at 2.5 ns on both
  starts.
- **Figure:** Fig. 3h,i.

### C7 — Backbone-agnostic and training-free

- **Evidence:** the backbone is frozen; the readout is fitted from a 100 ps reference ACF; the
  readout adds negligible cost; the same coefficients work from two initial conditions.
- **Warrant:** no backbone gradients and no new labels; the method is a post-hoc readout.
- **Qualifier:** only one backbone family tested; the readout still needs a reference NVT
  trajectory. "Training-free" means no gradient training, not data-free.
- **Falsifier:** applying the same readout to a second backbone and failing to improve its
  partition/structure would limit the agnosticism claim.
- **Figure:** Fig. 5, Methods.

---

## 4. Section-level logical flow

### Introduction

1. Direct force-free propagators promise large speedups (TrajCast existence proof).
2. The field evaluates them with one-step MAE and short rollouts; both can mislead.
3. The real target is the long-time stationary measure.
4. Existing remedies: global thermostats, structure-preserving action maps, learned velocity
   updates, GLE thermostats, stochastic/generative propagators. Each has a gap.
5. Our move: separate structure (backbone) from thermodynamics (mode-resolved readout).
6. Contributions (diagnostic, architecture, guarantee, water results).

### Results 1 — The diagnostic

- Present the pointwise-vs-ensemble reversal first, as a puzzle.
- Then the NVE red herring.
- Then the mode partition table, which resolves the puzzle: the failing models are not unstable;
  they are partitioned wrongly.
- End with the design requirement: control the mode-resolved stationary measure.

### Results 2 — The method

- Start from the requirement, not from the GLE literature: we need per-mode variance and
  per-mode memory.
- Derive the AR(p) + FD noise + rescale.
- State exactly what is guaranteed and what is not.
- Emphasize the decoupling: positions from the backbone, velocities from the readout.

### Results 3 — Water

- 100 ps two starts: partition, ACF, RDF, VDOS.
- 2.5 ns: diffusion and stability.
- Compare to published checkpoint on the same protocol.
- Show the GLR beats the published model on structure/VDOS and matches diffusion.

### Results 4 — Ablations

- p=1 → p=10 → p=20: memory order controls O–H/VDOS and the translation tail.
- Rescale off: partition collapses.
- FD noise off: fluctuations vanish.
- Internal blend 1.0: single-gain spectrum failure.
- Seeds: error bars.

### Discussion

- The diagnostic generalizes: evaluate ensemble-level, mode-resolved.
- The readout is a general post-hoc fix for force-free propagators.
- Relation to GLE thermostats: same physics, different object (a readout on a learned map, with a
  mode-resolved guarantee and a diagnostic).
- Relation to action maps: deterministic structure preservation vs stochastic canonical sampling.
- Limitations: kinetic guarantee only, reference NVT data, one system, internal long tail,
  long-range physics.

---

## 5. Counter-arguments and responses

### "This is just a GLE thermostat on a neural network."

**Response:** A GLE thermostat acts on forces/velocities in an MD integrator; here there is no
force and no integrator. The GLR is a readout on a frozen learned propagator: positions come from
the backbone, velocities from the readout, and the readout is fitted from the reference ACF, not
from a physical memory kernel. The contribution is the demonstration that this decoupling fixes
the failure mode of force-free propagators, plus the diagnostic that identifies the failure. The
ablations (AR order, rescale, noise) show the readout is doing the work, not a generic thermostat.

### "Bigi/Ceriotti already fix equipartition with action maps."

**Response:** Their correction is a deterministic, structure-preserving (symplectic,
time-reversible) map targeting energy/equipartition. Ours is a stochastic, mode-resolved
canonical sampler targeting the NVT mode partition, RDF, VDOS, and diffusion. They are
complementary: an action map could be combined with a GLR readout. We should cite and compare
conceptually, and run their method on water if code is available.

### "MAMD already does learned velocity updates for stable MD."

**Response:** MAMD uses a black-box LSTM with periodic injection; it targets stability, not the
equilibrium mode partition. The GLR uses interpretable AR(p) coefficients from the reference ACF,
FD-consistent noise, and exact per-mode rescaling; it guarantees the partition and matches
multi-lag ACFs. The O–H/VDOS and diffusion results are the empirical delta.

### "TrajCast already passes; why is this needed?"

**Response:** TrajCast is an existence proof, not a guarantee. Our diagnostic shows a
pointwise-better model that fails, and the published model itself passes only because its
particular bias happens to be small. The GLR makes the partition a design constraint, is
backbone-agnostic, and beats the published checkpoint on O–O/O–H/H–H RDF and VDOS in our
analysis.

### "The guarantee is only kinetic, not the full canonical ensemble."

**Response:** Concede explicitly. The readout guarantees the **mean** mode kinetic partition and
the velocity ACF; the per-step rescale suppresses the mode-KE fluctuations, and the position
marginal is learned and validated empirically. This is the honest scope, and it is sufficient to
fix the observed failure. A full joint guarantee—including the correct kinetic-energy
fluctuations—would require a stochastic rescale, an energy-based sampler, or MCMC, which we do not
claim.

### "Positions and velocities are inconsistent."

**Response:** The scheme is intentionally decoupled: the backbone supplies the displacement, the
readout supplies the velocity variable used for the next backbone call and for the kinetic
diagnostics. The readout is not integrated as a second position update. This avoids a double
integrator and keeps the learned displacement in distribution; the empirical RDF/VDOS/diffusion
show the composition is consistent at the ensemble level. We tested the obvious "fix"—adding
\(\Delta t(v^{\rm GLR}-v^{\rm pred})\) to the displacement—and it crashed at 24.8 ps, so the
decoupling is a stability requirement, not an arbitrary choice.

### "You need reference NVT data."

**Response:** Yes, a short (100 ps) reference NVT trajectory is needed to fit the ACF. This is
much cheaper than retraining and is available for most systems of interest. We state it as a
requirement, not a hidden assumption. A future direction is to fit the memory kernel from NVE or
from the backbone itself.

### "Single system, single seed."

**Response:** The paper is a water method paper; we will add ≥3 seeds for the final configuration
and frame the claim as architectural, not universal. A second system is future work unless the
quartz data can be adapted.

---

## 6. Novelty statement (precise)

Relative to TrajCast: same task class, but a **mode-resolved stationary-measure diagnostic** and a
**readout that guarantees the canonical partition**, with structure results beyond the published
checkpoint.

Relative to Bigi/Ceriotti: deterministic structure-preserving action map vs stochastic
mode-resolved canonical readout; energy/equipartition vs NVT partition + RDF/VDOS/diffusion.

Relative to MAMD: black-box periodic LSTM velocity injection vs interpretable AR(p) +
FD-consistent noise + exact per-mode rescale.

Relative to GLE thermostats: a thermostat for MD integrators vs a readout for a frozen force-free
propagator, fitted from the reference ACF, with a mode-resolved guarantee and a diagnostic.

Relative to stochastic/generative propagators: no MCMC, no energy function, no retraining; a
cheap post-hoc readout.

---

## 7. Falsification / gate conditions

| Claim | Gate | If it fails |
|---|---|---|
| C4 partition guarantee | rescale-on partition within finite-sample error on two starts | fix rescale/noise implementation |
| C5 structure | GLR RDF/VDOS better than published on ≥2 seeds | soften to "comparable"; report seed spread |
| C6 diffusion | D within 1.4–2.5e-9 on 2.5 ns, both starts | p=20 / internal memory; otherwise no diffusion claim |
| C6 independence | 2.5 ns independent reproduces partition/ACF/RDF | investigate initial-condition memory |
| C7 agnosticism | second backbone improves or at least does not collapse | scope to one backbone |

---

## 8. Title and abstract logic

**Title candidates:**

1. *Thermodynamically consistent force-free molecular dynamics via a mode-resolved generalized
   Langevin readout*
2. *Structure from the backbone, thermodynamics from the readout*
3. *A mode-resolved generalized-Langevin readout makes force-free propagators canonically
   consistent*

**Abstract order:** problem (force-free speed vs long-time fidelity) → diagnostic (pointwise/NVE
mislead; mode partition is the issue) → method (AR(p) + FD noise + rescale on a frozen backbone)
→ results (partition, ACF, RDF, VDOS, diffusion) → scope (backbone-agnostic, training-free,
water).
