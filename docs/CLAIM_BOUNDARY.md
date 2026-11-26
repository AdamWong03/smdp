# Claim boundary — backbone vs readout

**Date:** 2026-11-27  
**Purpose:** prevent the two project routes from being conflated in the paper, README, or talks.

## The two routes

| | Route A — raw force-free backbone | Route B — fixed GLR readout |
|---|---|---|
| Object | a trained direct propagator (relational or phase-space) | a statistical velocity readout on a **frozen** backbone |
| What it predicts | displacement and velocity from the current state | velocity correction from the reference mode ACF + variance target |
| Short-time evidence | water 5 fs pos RMSE 4.9e-4 Å; quartz 30 fs 1.9e-3 Å | applied to frozen checkpoints; no retraining |
| Long-time water | **fails**: v0 NVT VDOS 0.356/0.236, \(D=2.8\times10^{-13}\) m²/s | partition/RDF/VDOS restored; **\(D\) 26% low** |
| Long-time quartz | 30 fs NVE drift +0.904/−0.330 eV (inconsistent) | 5 fs, 100 ps, two seeds: \(T=299.9\) K, VDOS 0.950/0.955, Si–O RDF 0.028 |
| Method status | not a long-time method | **this is the paper's method** |

## What the paper claims

> A frozen force-free backbone supplies the displacement; a fixed mode-resolved GLR readout
> supplies the velocity and corrects the **stationary measure**. The readout is the method; the
> backbone is a plug-in.

The paper **does not** claim:

- that the raw phase-space or relational network alone reproduces long-time water dynamics;
- that the 30 fs quartz backbone is improved by the 5 fs phonon-band result;
- that the GLR recovers \(D\) exactly (water is 26% low);
- that the deterministic rescale reproduces the kinetic-energy variance (it is mean-only);
- that the stochastic rescale is free (it restores the KE variance but breaks GK/MSD consistency);
- that quartz is validated at 2.5 ns (only 100 ps, two seeds);
- that the new GLE readout improves autonomous dynamics (its offline gate failed);
- that FDT alone guarantees Gibbs invariance when composed with frozen R3-Opt (the composed
  process was not validated);
- that the 100 ps official-TrajCast control is the 2.5 ns result (the 2.5 ns run is pending).

## What is locked

- The readout algorithm: physics-defined mode basis, AR(10) memory where the backbone ACF is
  wrong, group-resolved variance rescale, blend \(\lambda=0.1\), COM projection,
  \(q'=q+\Delta q\).
- Water production: molecular modes + memory + deterministic rescale (GK/MSD consistent).
- Quartz: phonon-band partition + stochastic rescale + memory off.
- The negative controls: full species GLR, full phonon GLR, per-mode rescale.

## What is not locked

- Whether the 26% water \(D\) deficit is a readout failure or a calibration-window bias. The
  25×100 ps reference ACF integral is 14.45 ± 1.86 fs; the production GLR used one 100 ps block
  and a 200 fs window. A full 2.5 ns mean and a D-constrained refit are needed.
- Whether a single rescale can satisfy both the KE marginal and GK/MSD consistency.
- Quartz at 2.5 ns (Stage 1).
- Temperature transfer, larger box, third material, action-map baseline (Stage 2).

## GLE gate (2026-11-27)

The bounded 4–6 mode DHO+slow GLE study was executed and **failed the offline gate**:

- The exact 2×2 complex-OU realization is correct (Monte Carlo ACF RMSE 0.0025).
- The unconstrained 3-mode fit matches the 5/25 fs ACF but its discrete D is **1.90×** the
  reference; the unconstrained 6-mode is 1.52×. The near-Nyquist ω≈1.25 rad/fs mode is aliased
  at the 5 fs cadence.
- The frequency-bounded 5-mode fit restores D (+1.0%) and the long-lag ACF (RMSE 0.047) but its
  5 fs ACF is 0.726 vs 0.982 (error 0.255); the 6-mode does not fix it.
- The independent `train_nvt_200ps` reference shows the same 5/25 fs shape, so this is not a
  calibration-block artifact.

**Classification:** kernel identification at 5 fs, not discrete implementation, not neural
coupling. No 100 ps or 2.5 ns GLE rollout was started. A different kernel family or a 1 fs
cadence would be needed; neither is authorized.

## Official TrajCast 2.5 ns (pending)

The official IBM published checkpoint is running under the same initial state, CSVR thermostat,
5 fs horizon, and 50 ps thermalization + 2.5 ns production protocol. The 100 ps control has
excellent temperature/total-KE fluctuations and O VDOS 0.937, with H VDOS 0.624 and ~5–6%
mode-partition differences. Do not report a 2.5 ns number until
`experiments/published_trajcast_2p5ns_20261126/analysis/trajcast_2p5ns_report.json` exists.

## Version-control state

- Local reproducible snapshot: `reproducibility/snapshot_20261127/`, git branch
  `research/glr-gle-20261127`, tag `research-snapshot-20261127` (local-only; not pushed).
- GitHub branch `research/h1h2-selection-20261120` still stops at `4399ccd`; the current paper,
  experiments, and GLR coefficients are local-only.
- The snapshot includes the GLR/GLE code, coefficients, fit/audit results, logs, `MANIFEST.json`,
  and `CHECKSUMS.sha256`. Large trajectories are referenced, not copied.
- README and RUNNING_JOBS were stale; this claim boundary and `SMDP_ICML2027_STATUS_AUDIT.md`
  are the authoritative documents.

**Action:** push the snapshot branch/tag and freeze a submission tag only after the official
TrajCast 2.5 ns result is analyzed and the GLR provenance/refit decision is recorded.
