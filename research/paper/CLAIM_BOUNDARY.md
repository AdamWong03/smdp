# Claim boundary — backbone vs readout

**Date:** 2026-11-23  
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
- that quartz is validated at 2.5 ns (only 100 ps, two seeds).

## What is locked

- The readout algorithm: physics-defined mode basis, AR(10) memory where the backbone ACF is
  wrong, group-resolved variance rescale, blend \(\lambda=0.1\), COM projection,
  \(q'=q+\Delta q\).
- Water production: molecular modes + memory + deterministic rescale (GK/MSD consistent).
- Quartz: phonon-band partition + stochastic rescale + memory off.
- The negative controls: full species GLR, full phonon GLR, per-mode rescale.

## What is not locked

- Whether a slow translation mode can close the 26% water \(D\) deficit without degrading RDF
  (2.5 ns gate running).
- Whether a single rescale can satisfy both the KE marginal and GK/MSD consistency.
- Quartz at 2.5 ns (Stage 1).
- Temperature transfer, larger box, third material, action-map baseline (Stage 2).

## Version-control state

- Branch: `research/h1h2-selection-20261120`, commit `1cffee5`.
- 2 modified tracked files, 29 untracked GLR/analysis files.
- GitHub branch stops at `61b258d`; the local branch is not pushed.
- README and RUNNING_JOBS were stale; the paper is the authoritative claim document.

**Action:** freeze a submission tag only after the untracked GLR code, coefficients, and analysis
scripts are committed and the README is updated. Do not rent GPUs against an uncommitted method.
