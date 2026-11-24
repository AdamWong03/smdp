# Phase 1 — Novelty Matrix

**Date:** 2026-11-22
**Status:** focused gap check; two close works require explicit differentiation.

---

## 1. Closest prior art

### A. TrajCast (Thiemann et al., Nat. Mach. Intell. 8, 764, 2026)
- Direct, force-free, autoregressive 5 fs propagator; trained on NVE; global CSVR at inference.
- Establishes that direct propagators can reproduce water structure/VDOS/diffusion.
- **Overlap:** same task class (force-free direct propagator).
- **Delta:** no mode-resolved thermodynamics; no guarantee of the stationary partition; no
  diagnostic of pointwise-vs-ensemble decoupling. Our work explains why TrajCast-like models can
  fail and adds a readout that guarantees the partition.

### B. Bigi, Spies & Ceriotti, "Learning the action for long-time-step simulations of MD"
(Phys. Rev. Lett. 136, 237301 (2026); arXiv:2508.01068)
- Learns data-driven **structure-preserving (symplectic, time-reversible) maps** = the mechanical
  action; eliminates energy-conservation and **equipartition** artifacts of non-structure-preserving
  ML predictors.
- Can be applied iteratively as a **correction to cheaper direct predictors**.
- **Overlap:** long-time stability/equipartition; correcting a direct predictor; short reference
  trajectories; transfer across conditions.
- **Delta:** their correction is an **action/symplectic map** (deterministic, structure-preserving,
  action/force-based). Ours is a **force-free stochastic mode-resolved GLE readout** with
  FD-consistent noise and exact per-mode variance rescaling; it targets the **canonical NVT
  ensemble** (partition, RDF, VDOS, diffusion), not only energy/equipartition. The diagnostic
  (pointwise accuracy is not predictive) is also absent there.
- **Risk:** reviewers may see "correct a direct predictor with a physically constrained map" as
  overlapping. The paper must compare the two corrections and show the mode-resolved NVT advantage.

### C. Dayhoff & Varma, "Velocity updates from machine learning for stable MD integration"
(AIP Advances 16, 015205, 2026; ChemRxiv 2025)
- **MAMD:** stacked LSTM velocity updates from historical velocity/coordinate data; periodic,
  infrequent injections; force-free coordinate propagation; recovers MD stability.
- **Overlap:** learned memory-based velocity updates; force-free; long-time stability.
- **Delta:** MAMD is a black-box LSTM with periodic injection; ours is a **mode-resolved AR(p)**
  readout with **Yule–Walker coefficients from the reference ACF**, **FD-consistent noise**, and
  **per-mode variance rescaling**. Ours guarantees the canonical mode partition and matches
  multi-lag ACFs; MAMD targets stability, not the equilibrium mode partition.
- **Risk:** "learned velocity update with memory for stable MD" is the same high-level idea.
  The mode resolution + FD guarantee + structure results are the delta.

### D. GLE / colored-noise thermostats (Ceriotti, Bussi, Parrinello, PRL 2009; GLE4MD)
- Colored-noise GLE thermostats with memory kernels; frequency-dependent / mode-selective
  thermostats (e.g., the δ-thermostat; quantum ZPE thermostats for water, PRB 87, 134207).
- **Overlap:** memory kernel + FD noise + mode/frequency resolution; equipartition.
- **Delta:** these are **MD thermostats acting on forces/velocities**, not a readout on a learned
  force-free propagator. Our readout is fitted from the reference ACF, applied to a frozen neural
  propagator, and includes exact per-mode variance rescaling; no forces.
- **Risk:** a reviewer may call the GLR "a GLE thermostat." The paper must emphasize the
  **learned-propagator mode bias**, the **structure/ensemble results**, and the diagnostic.

### E. Mode/frequency-resolved equipartition in water MD
- Rigid-water MD requires small timesteps for translational/rotational equipartition
  (J. Chem. Theory Comput. 2024); mode-specific thermostats are standard.
- **Overlap:** mode-resolved equipartition.
- **Delta:** classical MD thermostat vs a readout on a learned force-free propagator; ours
  guarantees the canonical partition by construction and matches multi-lag ACFs.

### F. Stochastic/generative propagators (Timewarp, ITO, MDGen, LiFlow)
- Learn stochastic kernels / flows; Timewarp uses MCMC to guarantee Boltzmann; ITO learns
  multiple time resolutions; MDGen samples trajectories.
- **Overlap:** stochastic dynamics; equilibrium sampling.
- **Delta:** deterministic frozen backbone + interpretable mode-resolved GLE readout; no MCMC,
  no energy function, no retraining of the backbone; cheaper.

### G. Markovian embedding of the GLE
- AR(p)/auxiliary-variable representations of memory kernels (GLE4MD; Markovian embedding
  literature).
- **Overlap:** AR(p) memory representation.
- **Delta:** we use it as a **mode-resolved readout on a neural propagator**, with Yule–Walker
  coefficients from the reference ACF and exact variance rescaling; the application and the
  diagnostic are new.

---

## 2. Positioning statement

> **Pointwise one-step accuracy is neither necessary nor sufficient for long-time equilibrium.
> The failure of force-free propagators is a mode-resolved stationary-measure problem. A frozen
> backbone can supply the structure; a mode-resolved generalized-Langevin readout—AR(p) memory
> kernels fitted from the reference ACF, FD-consistent noise, and per-mode variance
> rescaling—guarantees the canonical partition and matches multi-lag dynamics without forces.**

Novel contributions:

1. **Diagnostic:** the pointwise-vs-ensemble decoupling and the NVE-energy red herring.
2. **Architecture:** the mode-resolved AR(p) memory readout with exact variance rescaling, applied
   to a frozen force-free propagator. The ablation shows the memory kernel and rescale are the
   load-bearing components; the FD noise is theoretically consistent but not empirically necessary
   at the small blend used here.
3. **Guarantee:** canonical mean mode partition and stability by construction, backbone-agnostic
   and training-free (readout fitted from the reference ACF).
4. **Results:** water 2.5 ns RDF/VDOS better than the published TrajCast checkpoint; diffusion is
   26% low on two starts and is reported as a limitation, not as exact recovery.

---

## 3. Required comparisons / citations

| Work | Why | Action |
|---|---|---|
| TrajCast (NMI 2026) | direct baseline; published checkpoint | cite + compare 100 ps/2.5 ns |
| Bigi/Ceriotti action map (2026) | closest structure-preserving correction | cite; compare conceptually; if code available, run on water |
| MAMD (AIP Advances 2026) | closest learned velocity-update correction | cite; compare conceptually; if code available, run |
| GLE thermostats / GLE4MD | physics prior art for memory kernel + FD noise | cite; distinguish |
| Mode/frequency thermostats | mode-resolved equipartition prior art | cite; distinguish |
| Timewarp / ITO / MDGen | stochastic/generative alternatives | cite; distinguish |
| Markovian embedding | AR(p) memory representation | cite; method background |

---

## 4. Novelty risks and mitigations

| Risk | Mitigation |
|---|---|
| "It's a GLE thermostat on a neural network." | Emphasize the learned-propagator mode bias, the diagnostic, the exact per-mode variance rescale, and the NVT structure results; show ablations (rescale off, FD noise off, AR(1) vs AR(p)). |
| "Bigi/Ceriotti already fix equipartition." | Distinguish action/symplectic (deterministic, energy/equipartition) from mode-resolved stochastic GLE (canonical NVT ensemble, RDF/VDOS/diffusion); compare if possible. |
| "MAMD already does learned velocity updates." | Distinguish black-box LSTM + periodic injection from interpretable mode-resolved AR(p) + FD noise + exact partition guarantee. |
| "TrajCast already works." | The diagnostic explains why it can fail; the GLR beats it on O–O/O–H/H–H RDF and VDOS; the method is backbone-agnostic. |
| "Only water." | Scope explicitly or add quartz. |
