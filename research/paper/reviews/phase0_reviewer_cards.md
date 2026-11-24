# Phase 0 — Reviewer Configuration Cards

**Manuscript:** *Thermodynamically consistent force-free molecular dynamics via a mode-resolved
generalized Langevin readout*
**Target:** Nature Machine Intelligence (method paper)
**Review mode:** full, 5 seats (4 card-backed + fixed Devil's Advocate)

## Coverage

| Item | Status |
|---|---|
| Full manuscript (`main.pdf`, 16 pages) | read |
| Figures 1–4, Table 1 | read |
| `EVIDENCE_MAP.md`, `DIFFUSION_GATE.md`, `NOVELTY_MATRIX.md` | read |
| 2.5 ns runs, ablations, seeds | read |
| Supplementary Information | outline only |
| Code/data package | not assembled |
| Authors/affiliations | missing |
| Cover letter / AI disclosure | not written |

## Field analysis

- Primary: machine learning for molecular simulation (ML-MD)
- Secondary: computational/physical chemistry; statistical mechanics
- Paradigm: method development + empirical benchmark
- Method type: force-free learned propagator + mode-resolved stochastic readout; NVT water
- Target: Nature Machine Intelligence, method paper
- Maturity: complete draft; one water model, one backbone; diffusion 26% low; no SI/code package

## Cards

### 1. Journal-Fit Reviewer (EIC)

NMI editor with an ML-for-science background. Focus: novelty beyond TrajCast, broad readership,
claim–evidence balance. Likely question: is this a conceptual advance or an incremental
thermostat, and why does it belong in NMI rather than a specialized simulation journal?

### 2. Methodology Reviewer (R1)

ML-MD methodologist/statistician. Focus: baselines, ablations, error bars, reproducibility, data
leakage (reference-ACF fit vs held-out observables), the 26% diffusion deficit, seed/sample size,
code/data availability.

### 3. Domain Reviewer (R2)

Computational/physical chemist working on water MD and GLE thermostats. Focus: SPC/Fw reference,
RDF/VDOS/diffusion, mode decomposition, prior art (TrajCast; Bigi–Ceriotti PRL; MAMD), physical
meaning of the mean-partition guarantee, internal ACF long tail.

### 4. Perspective Reviewer (R3)

Statistical mechanician/generative modeler. Focus: what “thermodynamic consistency” means when
mode-KE fluctuations are suppressed; decoupled position/velocity consistency; generalizability
beyond water; practical cost/reference-data requirement.

### 5. Devil’s Advocate (fixed)

Strongest counter-argument: the paper claims thermodynamic consistency, but the position dynamics
are not consistent with the velocity readout (D 26% low), the FD noise is not necessary at
λ = 0.1, and the method is a mode-resolved GLE thermostat fitted to the target ACF; the novelty
relative to Bigi–Ceriotti and MAMD is incremental. It will test whether “thermodynamically
consistent” is falsifiable and whether the claims are calibrated.

## Stakeholder coverage

- Beneficiaries: ML-MD developers; computational chemists wanting fast NVT water.
- Cost-bearers: users who must generate a 100 ps reference NVT trajectory; groups without such
  data; researchers needing transport coefficients.
- Left out: solids/phonons, long-range electrostatics, non-water liquids, NPT, temperature
  transfer.
