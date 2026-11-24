# Research state

**Project:** force-free learned propagators for long-time molecular dynamics  
**Date:** 2026-11-23  
**Entry:** existing project with partial results; `resume` → `frame`  
**Gate A:** pending human sign-off on `RESEARCH_CONTRACT.md`

## Objective / capability

A force-free learned propagator with credible long-time NVT MD: target stationary distribution including kinetic-energy fluctuations, structure/VDOS, time correlations and transport, stable and initial-condition independent at 2.5 ns. Not merely improved partial statistics, and not a relabeled partial correction.

## Current candidate

The GLR readout family on frozen backbones. There is **no single locked configuration** yet:

- water: molecular modes, AR(10), deterministic variance rescale; stochastic and slow-mode variants are ablations;
- quartz: 5 fs, phonon-band stochastic rescale, memory off;
- full species and full phonon memory are negative controls.

Under the Gate A **hard cross-material gate**, the current L1 configuration is **two configurations**, not one rule: water deterministic vs quartz stochastic, memory on vs off. It fails the gate by construction. A candidate that needs a per-system switch is not a candidate for this contract.

The next candidate has not been chosen. `design` must compare mechanisms and select one that can be evaluated on both water and quartz from the start; do not start from "GLE is the answer."

## Key assumptions

1. A local AR/slow-mode kernel fitted to the reference ACF can represent the composed system's stationary and time-dependent behavior.
2. The backbone displacement can supply the position marginal while the readout supplies the velocity.
3. A variance rescale can be made compatible with the ACF/D without breaking structure.
4. One rule can transfer across water and quartz with only the physics-defined basis changing.

Each assumption is a hypothesis, not an established result.

## Claims

| Claim | Status | Evidence | Boundary |
|---|---|---|---|
| Short-time accuracy does not imply long-time fidelity | **supported** | raw v0 water NVT VDOS 0.356/0.236, D 2.8e-13; quartz 30 fs NVE drift +0.904/−0.330 eV | two models; not a controlled correlation |
| GLR improves mean temperature/partition/RDF/VDOS on water | **supported** | deterministic 2.5 ns: RDF 0.031/0.038/0.028, VDOS 0.847/0.953, D 1.406/1.393; stochastic 2.5 ns: KE std 0.437/0.429, VDOS 0.852/0.949 | within tested 64-molecule, 300 K scope |
| GLR restores the target stationary distribution | **contradicted / unresolved** | deterministic total KE std 0.0097 vs 0.4398 eV; stochastic GK 1.16 vs MSD 1.44 ×10⁻⁹ | deterministic is mean-only; stochastic breaks GK/MSD |
| GLR recovers D | **contradicted** | 1.39–1.41 vs 1.88 ×10⁻⁹; 26% low | translation ACF/kernel limitation |
| Quartz 5 fs phonon-band stochastic rescale is positive | **supported at 100 ps** | two seeds: T 299.9, KE std 0.398, VDOS 0.950/0.955, Si–O RDF 0.028 | not 2.5 ns; no transport claim |
| Full phonon memory is positive | **contradicted** | VDOS 0.734/0.838; no-rescale 0.594/0.686 | 5 fs backbone ACF already correct |
| One rule transfers across water and quartz | **contradicted by current configuration** | water deterministic; quartz band stochastic; memory on/off differs | does not prove no rule exists; the current L1 is not one rule |
| Static covariance matching fixes the KE variance | **contradicted** | `covmatch-offline-20261123`: covariance matched, KE std 0.105 vs ref 0.240; GK 1.445 vs 1.502 (100 ps) | KE variance is temporal; static covariance is not enough |
| 5 fs displacement defines a reliable instantaneous drift for a self-consistent GLE | **contradicted / high-risk** | `gle-interface-20261123`: \(a_{\rm fd}\) vs \(a_t\) corr 0.54, RMSE/std 0.87; \(\Delta v\) vs impulse corr 0.036 | 5 fs is an interval average, not an instantaneous acceleration |
| 1 fs displacement defines a reliable instantaneous drift for a position-consistent GLE | **supported (feasibility)** | `interface-resolution-20261123`: corr 0.987, RMSE/std 0.162; \(\Delta v\) vs impulse 0.948 | 1 fs only; a 1 fs backbone and GLE integrator are not yet built |
| 2 fs displacement defines a reliable instantaneous drift | **marginal** | `interface-resolution-20261123`: corr 0.920; \(\Delta v\) vs impulse 0.798 | speed/accuracy compromise; not chosen |
| A 1 fs backbone + position-consistent GLE can satisfy the water frontier gate | **hypothesis** | interface corr 0.987; 1 fs backbone test pos RMSE 3.16e-5 Å, vel 4.58e-5 Å/fs | backbone ready; GLE not yet built/tested; speedup ~2× |
| AR(50) memory + backbone drift is a stable position-consistent GLE | **contradicted** | `water-1fs-driven-gle-smoke`: preT 3509 K, min 0.50 Å at 4 steps | AR predictor is not a passive memory kernel |
| Langevin drift + global CSVR is a valid position-consistent method | **contradicted** | `water-1fs-langevin-csVR-100ps`: T 301.47, but partition 3.04/2.90/1.50 vs 2.37/2.43/2.60; RDF O–H 0.862; VDOS 0.520/0.442; D 4.13e-10 | total KE fixed, mode partition/structure destroyed |
| A post-hoc PSD GLE kernel fits the 1 fs ACF | **contradicted** | `velocity-gle-fit-1fs`: 5 fs ACF 0.315/0.346 vs 0.981; RMSE 0.071/0.066 | PSD kernel cannot fit short decay + long dip |
| Slow-mode + stochastic group rescale satisfies the water frontier gate | **near-miss / unsupported** | `slow-stoch-100ps-20261123`: KE std 0.4405; VDOS 0.858/0.970; D 1.480 (21.4% low); GK 1.386 vs MSD 1.480; O–H 0.1253 | D and O–H just outside gate; slow mode is water-specific, no quartz counterpart |
| 3-mode damped-oscillator GLE fits the translation ACF with D | **contradicted / limited** | `velocity-gle-fit-20261123`: integral 11.29 vs 11.34; 200 fs −0.041 vs −0.080; 500 fs +0.006 vs −0.022 | 3 modes too rigid; missing negative dip and long tail |
| AR(10)+slow, without post-hoc rescale, holds the composed-system temperature | **contradicted** | `slow-norescale-100ps-20261123`: T=70.1 K at 1k steps | local FD noise cannot control the composed system at blend 0.1 |
| Deterministic group mean + global CSVR satisfies the water frontier gate | **contradicted** | `slow-global-100ps-20261123`: total KE std 0.434, but mode KE std 0.146/0.155/0.148 vs 0.239/0.234/0.250; D 1.246; GK 1.650 | global CSVR fixes total variance, not mode-resolved variance; GK/MSD inconsistent |
| The decoupled readout + post-hoc rescale family satisfies the water frontier gate | **contradicted** | `slow-stoch`, `slow-global`, `slow-weak-ar1`: none simultaneously meets mode KE std, D, GK/MSD, RDF | tension is structural at blend 0.1; stop τ/weight sweeps |
| Current method is ready for an NMI method paper | **not supported** | claim table above | paper gate not passed |

## Success criterion

See `RESEARCH_CONTRACT.md`. **Frontier-level**: at least the strongest published/available baseline on structure, VDOS, and transport within noise, and better on the KE distribution / stationary measure. Water gate: 2.5 ns, two starts/seeds, one fixed configuration, KE distribution + ACF/VDOS + D + GK/MSD consistency. Hard cross-material gate: quartz 5 fs, same rule, 100 ps/2.5 ns, VDOS/RDF/KE. Failure: revise or rebuild; only a method paper, no diagnostic/benchmark substitute.

## Gate A decisions (2026-11-23)

- Objective: credible long-time MD — confirmed.
- Hard cross-material gate: one rule, only the physics basis changes — confirmed.
- Success: frontier-level, operationalized in the contract.
- Failure: revise/rebuild; only a method paper; budget exhaustion returns to the human.
- Budget and decision rights: default.

## Target changes

| Date | Old objective | New objective | Reason | Human approved? |
|---|---|---|---|---|
| 2026-11-21/22 | trustworthy long-time MD | stationary-measure correction / NMI draft | partial GLR gains; D/KE conflicts unresolved | **no** — drift, not an approved target change |
| 2026-11-23 | stationary-measure correction | trustworthy long-time MD | project audit; restore original capability | **yes** (Gate A item 1) |
| 2026-11-23 | water and quartz as two configurations | one rule, only the physics basis changes (hard gate) | Gate A decision; current L1 fails by construction | **yes** (Gate A item 3) |

## Next decisive experiment

Candidate 1 (covariance-matching readout) was rejected by `covmatch-offline-20261123`: static covariance matched, but KE std stayed at 0.105 vs reference 0.240 and GK did not improve.

The 5 fs position-consistent GLE was also rejected by `gle-interface-20261123`: the finite-difference displacement does not define a reliable instantaneous acceleration (corr 0.54; \(\Delta v\) vs impulse corr 0.036). Do not implement that integrator at 5 fs.

Next test:

**Gate B decided: 1 fs; backbone complete.** `water-1fs-backbone-20261123` passed (test pos RMSE 3.16e-5 Å, vel 4.58e-5 Å/fs, epoch 393). Next:

**Structural conclusion:** the frozen-backbone + post-hoc readout/integrator family has failed the hard gate at both 5 fs and 1 fs. The 1 fs backbone is accurate (3.16e-5 Å), but AR-as-GLE is unstable, Langevin+CSVR destroys the mode partition/structure, and a post-hoc PSD kernel cannot fit the 1 fs ACF.

**Gate B decision needed:** choose one:

1. **End-to-end training with the integrator in the loop.** Train the 1 fs backbone so its drift is consistent with the GLE/thermostat and the long-time stationary distribution. This changes the frozen-backbone thesis but is the remaining principled method path. Cost: 1–2 weeks, GPU.
2. **Stop/pivot.** The current approach cannot produce a method paper under the hard gate; a diagnostic/benchmark paper requires a target change (previously rejected).
3. **Different architecture.** A learned force/acceleration or a stochastic flow map with a built-in invariant distribution; a new method project.

Do not add more post-hoc rescue kernels.

The speed advantage is ~2× vs 0.5 fs MD, not ~10× at 5 fs. Do not run more τ/weight sweeps on the decoupled readout; that family is exhausted.

A candidate that requires deterministic water + stochastic quartz, or memory on/off per system, is out of scope for the hard gate.

## Open questions

- Can a **two-level scheme** (deterministic spectral correction + global CSVR) restore the KE variance **and** the ACF/D without the per-mode stochastic rescale?
- Can a velocity-only GLE restore the KE variance **and** the ACF/D without the position-consistent drift interface?
- Can a single kernel/rescale satisfy the KE marginal **and** the ACF/D?
- Is the decoupled position/velocity update the fundamental limit?
- **Can one rule transfer to quartz with only the basis changing? This is now the central hard gate.**
- What fixed rule maps the basis to rescale groups without a per-system switch (e.g., a group-size criterion, not "species for quartz, molecular modes for water")?
- What is the minimal adequate test, and what budget does it need?
- What result would falsify the GLR route rather than just another variant?

## Key artifacts

- `RESEARCH_CONTRACT.md` — Gate A.
- `CANDIDATE.md` — candidate comparison and the covariance-matching test.
- `EXPERIMENTS.md` — run registry (first entry: covariance matching).
- `DECISION.md` — decision records.
- `paper/CLAIM_BOUNDARY.md` — claim boundary.
- `paper/DIFFUSION_GATE.md` — D/GK/kernel limitation.
- `method_development/CURRENT_METHOD_STATUS.md` — full experimental record.
- `paper/latex/main_clean.pdf` — current draft (experimental record, not submission).
- Water 2.5 ns: `experiments/glr_2p5ns_20261122`, `experiments/glr_independent_2p5ns_20261122`, stochastic counterparts.
- Quartz 5 fs: `experiments/quartz5_phonon_bands_rescale_only_100ps_20261123` and seed 2.
- Covariance test: `experiments/covariance_match_20261123_v2.json`.

## Gate A questions for the human

1. Is the objective **credible long-time MD**, not partial-statistics correction?
2. Are the success criteria and tolerances right?
3. Is the budget / decision-rights split right?
4. Is the stop condition right: no single rule → negative, no method paper?
5. Is the current claim boundary accurate and complete?
