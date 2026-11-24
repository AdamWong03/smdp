# NVE Energy-Stability Diagnostic — 2026-11-21

**Purpose:** test whether NVE energy stability explains the long-time NVT failure.
**Result:** it does **not**. The published model, which passes NVT, is the *least* NVE-stable
of the three. The energy-drift-penalty hypothesis is falsified.

Protocol: 25 ps NVE (no thermostat), 5 fs horizon, same independent NVT frame 0 (301.8 K).

---

## 1. Temperature traces

| Model | start T | 25 ps end T | min / max | verdict |
|---|---:|---:|---:|---|
| Published TrajCast | 301.8 K | **475.6 K** | 272.6 / **545.7** | heats strongly in NVE; **NVT passes** |
| M0 5k | 301.8 K | 387.3 K | 277 / 484 | heats; NVT fails |
| Base 250k | 301.8 K | 270.5 K | 222 / 361 | most NVE-stable; **NVT fails** |

Published trace: 301.8 → 330.9 (2.5 ps) → 307.3 (5 ps) → 356.4 (7.5 ps) → 390 (10 ps) →
396 (15 ps) → 447.9 (17.5 ps) → 475.6 (25 ps).
M0 trace: 301.8 → 422.9 (2.5 ps) → 384.5 (5 ps) → 416 (10 ps) → 436 (15 ps) → 387 (25 ps).
Base trace: 301.8 → 318.2 (5 ps) → 315.3 (10 ps) → 273.9 (15 ps) → 306.8 (20 ps) → 270.5 (25 ps).

## 2. What the control rules out

- **NVE energy stability is not the differentiator.** The published model heats more than M0 and
  the base in 25 ps NVE, yet its NVT partition is the only correct one. The base is the most
  NVE-stable and its NVT partition is the worst (internal 6.04).
- **The energy-drift penalty is falsified as a method.** Adding `λ_E · L_energy_drift` would
  penalize the published model, which is the only working model. Do not add it.
- **Double thermostatting is not the primary cause.** M0 heats without CSVR, but so does the
  published model. The NVE diagnostic does not explain the NVT failure.
- The **M0 NVT instability** is still real: the independent NVT start failed at 130 ps with
  preT 3798.7 K. The published NVT is stable at 300 K. This is an NVT failure, not an NVE one.

## 3. What the evidence actually supports

The only validated target is the **NVT stationary measure under the actual CSVR protocol**:

- Published: NVE heats, CSVR removes the energy, **partition correct** (2.505/2.463/2.444).
- Base: NVE relatively stable, CSVR composition gives **internal-hot** partition
  (0.538/0.786/6.044).
- M0: NVE heats, CSVR composition gives **internal-cold** partition and eventually unstable
  (3.41/3.21/1.01; failed at 130 ps on the independent start).

The relevant property is the **mode-resolved response of the model + CSVR map**, not NVE energy
conservation. The next method should therefore be a **differentiable CSVR rollout + ensemble
moment loss**, not an NVE energy penalty.

## 4. Paper positioning (still open)

The working claim is **ensemble fidelity under the evaluation protocol**, not NVE stability:

> **Long-time NVT fidelity is controlled by the stationary measure of the composed
> propagator + CSVR map. Neither one-step accuracy, NVT-pair MSE, nor NVE energy conservation
> is sufficient; the objective must target the composed NVT ensemble.**

Energy stability is a red herring for this failure mode: the published model is NVE-unstable and
NVT-correct.

## 5. Next actions

1. Implement a **differentiable short CSVR rollout** (K = 20–50 steps) from base 250k.
2. Loss: mode partition + temperature (+ optional RDF/VDOS moments) on the rolled-out NVT
   states. No NVE energy term.
3. Evaluate 100 ps NVT partition/RDF/VDOS; then the independent start; then 2.5 ns.
4. Do **not** add an NVE energy-drift penalty.
5. Lock the paper only if the composed NVT ensemble passes, including an independent start and
   2.5 ns.
