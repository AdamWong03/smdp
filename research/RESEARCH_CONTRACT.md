# Research contract (Gate A)

**Project:** force-free learned propagators for long-time molecular dynamics  
**Date:** 2026-11-23  
**Status:** draft for human sign-off

## Objective / capability

Develop a force-free learned propagator that reproduces **credible long-time MD** of condensed-phase water, and at least one solid, under NVT conditions:

- the target stationary distribution, including the kinetic-energy distribution and its fluctuations, not only the mean;
- structure and vibrational spectra (RDF, VDOS);
- time-correlation and transport observables (velocity ACF, diffusion, Green–Kubo/MSD consistency);
- stable, initial-condition-independent behavior at 2.5 ns.

The objective is **not** merely to improve a few long-time statistics, and not to relabel a partial correction as a solved MD method.

## Why now

Short-time force-free propagators already fit held-out transitions well, but the raw models fail long-time NVT. The GLR readout produces real partial improvements (temperature, mean partition, RDF, VDOS) but has unresolved conflicts between the kinetic-energy distribution and the dynamics. The project needs a decision about whether a self-consistent method is achievable, not another paper draft.

## Success criteria

### Frontier criterion

"Frontier" means: on the same system and protocol, the unified rule is at least as good as the strongest published or available baseline on structure, VDOS, and transport within noise, and better on the core gap (the kinetic-energy distribution / stationary measure). If no published baseline exists, the reference MD is the target. The numbers below are the default operationalization; they do not assume the current method passes.

### Water gate

On at least two starts/seeds, 2.5 ns, one fixed method configuration:

- temperature \(300\pm6\) K;
- mean mode partition within 0.05 eV of the reference;
- total and mode kinetic-energy standard deviations within 10% of the reference;
- RDF/VDOS at least the strongest published baseline: default O–H ≤0.06, O–O ≤0.05, H–H ≤0.05; O VDOS ≥0.94, H VDOS ≥0.80;
- translation ACF at 5/25/100/500 fs within 0.05 absolute; the internal 500 fs tail is reported as a limitation unless a candidate fixes it;
- diffusion absolute error ≤20% of the reference, and Green–Kubo/MSD consistent within 2σ;
- no instability, collapse, or hidden per-seed configuration change.

### Hard cross-material gate (quartz)

The **same rule** must pass on quartz at 5 fs, changing only the physics-defined mode basis:

- 100 ps and 2.5 ns, two seeds;
- \(T=300\pm3\) K, stable;
- VDOS overlap ≥0.90;
- Si–O RDF RMSE ≤0.05.

"Same rule" means:

- same memory policy and hyperparameters;
- same rescale target and partition logic (the partition may be induced by the basis, but the rule mapping basis → groups is fixed);
- same blend, centre-of-mass projection, and position update;
- any diagnostic that selects a component must be fixed before seeing quartz and tested prospectively.

No per-system switch between deterministic/stochastic rescale, memory on/off, species/band partition, or slow-mode on/off.

If the rule passes water but not quartz, the cross-material hard gate fails. Water-only is a scope reduction and requires a recorded target change and human approval. Do not claim a universal method.

### Failure / rebuild

A failed candidate is not a failed project. Revise or rebuild the method within the approved budget; do not substitute a diagnostic/benchmark paper. Only a method paper is an acceptable deliverable.

If a candidate passes water but fails the hard cross-material gate, the universal claim fails. Water-only is a target change and requires explicit human approval; it must not be presented as one fixed rule across materials.

If the approved budget is exhausted without a passing method, stop and return the decision to the human. Do not extend the budget, rename the objective, or submit a partial-statistics paper automatically.

## Claim boundary

**Currently supported:**

- Short-time accuracy does not imply long-time fidelity; the raw backbones fail NVT.
- The GLR improves some long-time statistics (temperature, mean partition, RDF, VDOS) within tested scopes.
- Quartz 5 fs phonon-band stochastic rescale is positive at 100 ps, two seeds (VDOS 0.950/0.955, Si–O RDF 0.028).
- Full species and full phonon memory are negative controls.

**Not currently supported:**

- restoration of the target stationary distribution (deterministic KE std 0.0097 vs 0.4398 eV; stochastic GK 1.16 vs MSD 1.44 ×10⁻⁹);
- credible long-time MD or exact transport (D 26% low);
- one rule across water and quartz: the current configuration is **not** a single rule (water deterministic, quartz band stochastic; memory on/off differs), so it fails the hard gate by construction.
- NMI readiness or a locked positive method claim.

## Scope

- Water: 64 molecules, two backbones (250k relational, v0 phase-space), 2.5 ns.
- Quartz: 162 atoms, 5 fs horizon, 100 ps at present.
- Not yet: temperature transfer, larger boxes, third material, foundation backbone, action-map baseline.

## Budget and resources

- Existing GPUs for bounded feasibility tests.
- Up to 4×A100/H100 for 1–2 days after a gate passes; larger rentals require a new decision.
- No fixed deadline; human time for Gates A/C.

## Decision rights

### AI may decide alone

- implementation, configuration, debugging;
- experiment scheduling within the approved budget;
- analysis, registry updates, and drafting;
- routine design iterations that do not change the object, capability, or budget.

### Requires human sign-off

- changing the objective, capability, or success criteria;
- changing the object of study;
- exceeding the approved budget;
- promoting a claim from unresolved to supported;
- entering the paper workflow;
- submission, authorship, AI disclosure, ethics.

## Escalation triggers

- target or capability change;
- budget overrun;
- claim status change;
- a result that would change the paper's central claim;
- safety, ethics, legal, or authorship issue;
- external dependency that blocks progress.

## Stop conditions

- No candidate passes the kinetic-energy + dynamics gate within a bounded sprint.
- The decoupled position/velocity architecture is shown to be the fundamental limit.
- The evidence cannot distinguish the candidate from a fair control.

In these cases, report the negative result and the unresolved decision; do not substitute a paper narrative.

## Human gates

- **Gate A (frame)**: this contract.
- **Gate B (candidate/target)**: only if the candidate changes the object, capability, or budget.
- **Gate C (claim)**: claim table in `RESEARCH.md`; human sign-off before paper.
- **Gate D (submission)**: final claims, limitations, references, authorship, disclosure.

## Gate A decisions (2026-11-23)

- Objective: credible long-time MD — confirmed.
- Hard cross-material gate: one rule, only the physics basis changes — confirmed.
- Success: frontier-level, operationalized by the criteria above.
- Failure: revise or rebuild; only a method paper; no diagnostic/benchmark substitute.
- Budget and decision rights: default as written.

## Gate B decision (2026-11-23)

- Architecture: 1 fs backbone + position-consistent GLE.
- Reason: the 5 fs finite-difference drift fails (corr 0.54; \(\Delta v\)/impulse 0.036); 1 fs is credible (corr 0.987).
- Trade-off: speed advantage is ~2× vs 0.5 fs MD, not ~10× at 5 fs.
- Budget: 1–2 days GPU for the water 1 fs backbone; quartz only after the water GLE gate passes.
- The 5 fs decoupled readout remains a negative/contrast result; do not submit it as a method paper.

## Target-change protocol

Record the old objective, new objective, reason, and human decision. Do not inherit the old success criterion under a new name.

## Sign-off

- Human:
- Date:
- Version: draft-1
