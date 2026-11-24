# Phase 2 — Paper Outline

**Date:** 2026-11-22
**Target:** Nature Machine Intelligence, method paper
**Working title:** *Thermodynamically consistent force-free molecular dynamics via a
mode-resolved generalized Langevin readout*

Alternative titles:

1. *Mode-resolved fluctuation–dissipation readout for long-time stable force-free MD*
2. *Structure from the backbone, thermodynamics from the readout*
3. *A generalized-Langevin readout makes force-free propagators thermodynamically consistent*

---

## Abstract sketch

Force-free direct propagators can accelerate molecular dynamics by large factors, but their
long-time equilibrium statistics are unreliable. We show that pointwise one-step accuracy is
neither necessary nor sufficient for long-time fidelity: a published propagator with a large
pointwise error reproduces the canonical water ensemble, while a pointwise-accurate model fails.
The failure is a **mode-resolved stationary-measure** problem, not an NVE energy problem. We
introduce a **mode-resolved generalized Langevin readout (GLR)**: a frozen propagator supplies
the displacement, while per-mode AR(p) memory kernels fitted from the reference ACF, FD-consistent
noise, and exact per-mode variance rescaling enforce the canonical **mean** kinetic partition by
construction (the per-step rescale suppresses mode-KE fluctuations; see Limitations).
On water, the GLR matches multi-lag velocity correlations, reproduces the 2.5 ns RDFs
(0.031/0.038/0.028) and O VDOS (0.953) better than the published TrajCast checkpoint, and is
stable and initial-condition independent. The diffusion coefficient is 1.41e-9, within 25% of the
reference (1.84–1.88e-9) and far closer than the AR(1) readout (3.59e-9); exact diffusion
recovery remains a limitation. The readout is backbone-agnostic, training-free, and force-free.

---

## 1. Introduction (~800 words)

- Force-free finite-time propagators: speed, data efficiency, TrajCast existence proof.
- The open failure: long-time equilibrium/partition; pointwise MSE does not see it.
- Existing fixes and their limits:
  - global CSVR (TrajCast) — no mode resolution/guarantee;
  - structure-preserving action maps (Bigi/Ceriotti 2026) — deterministic, energy/equipartition;
  - learned velocity updates (MAMD 2026) — black-box, periodic, stability;
  - GLE/mode thermostats — MD thermostats, not learned-propagator readouts;
  - stochastic/generative propagators (Timewarp/ITO/MDGen) — cost/complexity.
- Gap: a **mode-resolved, FD-consistent readout** that guarantees the canonical mean kinetic partition and
  the multi-lag dynamics of a frozen force-free propagator.
- Contributions (4).

## 2. Results (~2,200 words, 4 figures)

### 2.1 Pointwise accuracy is not the objective (Fig. 1)
- Published TrajCast: pointwise error 15× larger on our val transitions than the 250k model, yet
  its NVT partition is correct and the 250k fails.
- NVE energy is a red herring: published heats more in 25 ps NVE (302→476 K) than M0/base, yet
  passes NVT; base is the most NVE-stable and fails NVT.
- The failure is mode-resolved: partition table across models; M0 overshoot; CSRV blow-up.
- Claim C1/C2/C3.

### 2.2 Mode-resolved generalized Langevin readout (Fig. 2)
- Architecture: backbone displacement `q' = q + dq_pred`; velocity from GLR.
- Per-mode AR(p): `v_mode(t) = Σ a_k v_mode(t−k) + ε`; Yule–Walker from reference ACF.
- FD-consistent noise: prediction-error variance; target mode KE.
- Exact per-mode variance rescaling; COM/momentum projection.
- Blend λ with the backbone velocity to keep the position head in distribution.
- Algorithm box; hyperparameters (p=10, λ=0.1, gains from reference ACF).

### 2.3 Water NVT (Fig. 3)
- 100 ps, two initial conditions: partition, ACF, RDF, VDOS.
- 2.5 ns: diffusion, long-time stability.
- Comparison: TrajCast published checkpoint, FDR v1, M0, base 250k, reference.
- Key numbers: partition 2.48/2.53/2.40; O–O 0.032, O–H 0.104, H–H 0.026; VDOS 0.848/0.964 at
  100 ps; diffusion pending (gate).

### 2.4 Ablations (Fig. 4)
- AR(1) vs AR(10): O–H 0.301 → 0.104; diffusion tail.
- Internal blend 0.1 vs 1.0: 1.0 worse (0.957 O–H) — single-gain spectrum failure.
- Rescale off: variance collapse (FDR blend 0.1 cooled to 14 K).
- FD noise off / gain-only: no correct fluctuations.
- p = 1, 5, 10, 20; blend λ; seeds.
- Frozen backbone vs end-to-end: scope.

### 2.5 Scope and cost (optional Fig. 5 / table)
- ns/day; readout cost vs backbone.
- Water-only scope or quartz.
- Reference NVT data requirement (ACF fit).
- Long-range interactions not addressed.

## 3. Discussion (~700 words)

- Relation to GLE thermostats (Ceriotti), action maps (Bigi/Ceriotti), MAMD, TrajCast,
  stochastic/generative propagators.
- Why the diagnostic matters: evaluation should be ensemble-level, not one-step MAE.
- Limitations: internal ACF long tail; reference NVT data; water-only; single seed; long-range.
- Outlook: end-to-end training with the readout; other systems; temperature transfer.

## 4. Methods (~2,000 words)

- Data: SPC/Fw water, NVE training, NVT reference; independent NVT trajectory.
- Backbone: RelationalTransition 6 Å, 250k steps; published TrajCast checkpoint.
- GLR fit: mode decomposition, ACF, Yule–Walker, prediction error.
- Simulation: 5 fs, CSVR/FDR/GLR protocol, 100 ps/2.5 ns.
- Metrics: partition, ACF, RDF, VDOS, diffusion; blocks/error bars.
- Statistics: seeds, block analysis.

## 5. Figures

| Figure | Content |
|---|---|
| Fig. 1 | Pointwise-vs-ensemble diagnostic; NVE T traces; stability/partition table |
| Fig. 2 | GLR architecture + algorithm |
| Fig. 3 | Water NVT: partition, ACF, RDF, VDOS, MSD/diffusion; two starts |
| Fig. 4 | Ablations: AR order, internal blend, rescale, FD noise, seeds |
| Fig. 5 (optional) | Scope/cost; second system or water-only; limitations; omit if the four main figures suffice |

## 6. Claims

- C1: Pointwise one-step accuracy is neither necessary nor sufficient for long-time equilibrium.
- C2: NVE energy stability is not the differentiator; the composed NVT stationary measure is.
- C3: A mode-resolved AR(p) FD readout guarantees the canonical mean kinetic partition and matches multi-lag
  dynamics on a frozen force-free propagator.
- C4: The GLR reproduces water RDF/VDOS at or beyond the published TrajCast checkpoint; diffusion
  is 25% low and is reported as a limitation, not as exact recovery.
