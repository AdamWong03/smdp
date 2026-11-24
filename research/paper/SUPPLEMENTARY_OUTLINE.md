# Supplementary Information — Outline

**Target:** Nature Machine Intelligence, method paper.
**Status:** outline; fill after the diffusion gate and seed runs.

---

## Supplementary Notes

1. **Mode decomposition and kinetic partition.**
   - COM/rotation/internal definitions, orthogonality, degrees of freedom.
   - Why the decomposition is the right diagnostic for water.
   - Kurtosis check: 2.93/5.49/5.93 (GLR) vs 3.04/5.62/5.57 (reference).

2. **Yule–Walker fit and prediction errors.**
   - ACF estimator, Toeplitz system, prediction-error ratio.
   - p=1/5/10/20 coefficients and ACF comparison.
   - Stability of the AR polynomials (roots inside the unit circle).

3. **FD noise and variance rescaling.**
   - Derivation of the target noise variance.
   - Per-step rescale and COM projection.
   - What is guaranteed (mean partition, ACF) and what is not (KE fluctuations, joint distribution).

4. **Rollout protocol and safety stops.**
   - 5 fs horizon, no external thermostat, 0.65 Å / 3000 K safety stop.
   - Two initial conditions; 100 ps and 2.5 ns windows.
   - Published checkpoint native CSVR protocol.

5. **Diffusion analysis.**
   - Molecular COM, global drift removal, time-origin averaging.
   - Eight blocks, 50 fs cadence, 10–100 ps fit.
   - Reference D and block error.

6. **Ablations.**
   - AR order, blend, internal blend, rescale, FD noise, displacement coupling.
   - Seeds and error bars.
   - Frozen vs end-to-end (if available).

7. **Limitations.**
   - One water model, one backbone, local cutoff, reference-NVT data requirement.
   - Internal ACF long tail; suppressed KE fluctuations.
   - Long-range physics outside scope.

---

## Supplementary Figures

| Figure | Content | Source |
|---|---|---|
| S1 | Full translation/rotation/internal ACF (0–1000 fs) | `mode_gain/glr_*.json`, `reference.json` |
| S2 | AR order fits p=1/5/10/20 (all modes) | `glr_fit_p*_20261122` |
| S3 | Velocity-component distributions and kurtosis | trajectory analysis |
| S4 | RDF/VDOS for the independent start | `glr_independent_100ps_20261122/analysis` |
| S5 | Diffusion MSD blocks | `glr_2p5ns_20261122/analysis/msd.csv` |
| S6 | Seed-to-seed RDF/VDOS spread | seed runs |
| S7 | Stability/partition table for all controls | `MULTISTEP_STABILITY_TABLE_20261122.md` |

## Supplementary Tables

| Table | Content |
|---|---|
| S1 | Full per-method stability and partition table |
| S2 | Per-seed partition, RDF, VDOS, D |
| S3 | AR coefficients and prediction errors |
| S4 | Diffusion block fits (slope, intercept, R²) |

---

## Data and code availability

- Code: `/data2/wyd/research/foundation_mlip_adaptation/code/h1h2` (to be packaged).
- Reference and rollout trajectories: paths in `paper/EVIDENCE_MAP.md`.
- Published checkpoint: IBM Hugging Face `ibm-research/trajcast.models-arxiv2025`.
- No forces are used during propagation; the reference forces are used only for analysis.
