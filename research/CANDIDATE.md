# Candidate method brief

**Date:** 2026-11-23  
**Gate A objective:** credible long-time MD; one rule, only the physics basis changes.  
**Current state:** the deterministic GLR fixes the mean but suppresses the KE variance; the stochastic GLR restores the variance but breaks GK/MSD consistency. The current L1 configuration is two methods, not one.

## Candidate 1 — covariance-matching readout

### Mechanism

Keep the existing frozen-backbone + velocity-readout architecture, but replace the diagonal/stochastic variance rescale with a **constant linear transform** that matches the full mode velocity covariance:

\[
v' = A v,\qquad A = C_{\rm ref}^{1/2} C_{\rm model}^{-1/2},
\]

where \(C_{\rm model}\) is the velocity covariance of the blended/readout trajectory and \(C_{\rm ref}\) is the reference covariance in the same physics-defined mode basis.

If \(A\) is constant, it matches the stationary covariance (including the KE variance and cross-mode correlations) without post-hoc stochastic noise. A constant transform preserves the ACF up to the transform; whether the resulting ACF/D remains correct is an empirical question.

### Assumptions

- The mode covariance can be estimated reliably from a short baseline and the reference.
- \(C_{\rm model}\) is invertible or can be regularized without destroying the relevant modes.
- A constant transform does not destabilize the backbone feedback or the position update.
- The reference velocity covariance is the right target; matching it is not merely calibration on the same observable used to fit the kernel.

### Competing explanation

The gain may come from matching the static covariance, not from the memory kernel or the physics basis. The test must compare covariance matching against a covariance-only control and against the current deterministic/stochastic rescale.

### Discriminating experiment (first, offline)

Apply \(A\) to the existing deterministic water velocity trajectory in the molecular mode basis. Check:

- total and mode KE standard deviations;
- translation/rotation/internal ACF at 5/25/100/500 fs;
- Green–Kubo \(D\);
- condition number / regularization sensitivity.

If the transform cannot match the KE variance **and** preserve the ACF/D, stop before implementing it in the rollout.

### Cost and kill criterion

- Cost: CPU only, existing trajectories.
- Kill: ill-conditioned \(A\), KE variance not restored, or ACF/D degraded beyond the current deterministic result.

## Candidate 2 — fixed basis-induced rule (integration candidate)

### Mechanism

Use one fixed rule for both materials:

- build the physics-defined mode basis;
- group modes by a fixed basis-induced criterion (not "species for quartz, molecular modes for water");
- use the same stochastic group-rescale logic;
- use memory only when a fixed, pre-registered backbone-ACF diagnostic says it is needed;
- same blend, COM projection, position update.

### Assumptions

- A basis-induced grouping rule can produce groups with enough degrees of freedom on both water and quartz.
- The memory diagnostic is fixed before seeing quartz and tested prospectively.
- The rule does not require a per-system threshold.

### Discriminating experiment

Apply the same rule to water and quartz; check the hard cross-material gate. If it needs a per-system switch, it is not a candidate.

### Cost and kill criterion

- Cost: 100 ps runs on existing backbones.
- Kill: any per-system switch, or one material fails the gate.

## Candidate 3 — self-consistent GLE integrator

### Mechanism

\[
dq = v\,dt,\qquad
dv = a_{\rm backbone}(q,v)\,dt - \int_0^t K(t-s)v(s)\,ds\,dt + \eta(t)\,dt,
\]

with a PSD kernel and FD noise; positions integrate the same velocity.

### Assumptions

- The backbone's finite-time displacement can define a drift \(a_{\rm backbone}\) consistent with the GLE.
- The GLE's canonical stationary distribution survives the approximate, nonlinear backbone drift.
- The 5 fs discretization is stable.

### Discriminating experiment (before implementation)

Test whether the backbone-derived finite-difference acceleration approximates the instantaneous acceleration/force on held-out NVE data. If the interface error is too large, the GLE guarantee does not apply.

### Cost and kill criterion

- Cost: high; implementation and stability work.
- Kill: acceleration interface fails, or the composed system is unstable.

## Decision

Start with **Candidate 1**, because it directly targets the core conflict (KE variance vs ACF/GK–MSD) and can be tested offline on existing trajectories. If it passes, integrate it with Candidate 2's fixed basis-induced rule. Keep Candidate 3 only if 1 and 2 fail and the acceleration-interface test passes.

This is a design decision, not a claim that covariance matching will work.

## Update 2026-11-23 — first tests

- **Candidate 1 (covariance matching): rejected.** `covmatch-offline-20261123` matched the static covariance to ~1e-11 but left the translation KE std at 0.105 vs reference 0.240; ACF and GK did not improve. Static covariance is not enough.
- **Candidate 3 (position-consistent GLE at 5 fs): rejected before implementation.** `gle-interface-20261123` found \(a_{\rm fd}\) vs \(a_t\) correlation 0.54 and \(\Delta v\) vs impulse correlation 0.036. A 5 fs displacement is an interval average, not a reliable instantaneous acceleration.
- **Next candidate 2a: velocity-only GLE readout.** Fit a PSD velocity kernel + FD noise to the reference ACF; the backbone still supplies the displacement, so the acceleration interface is not needed. Standalone test: KE variance, ACF, D. If it passes, integrate with the frozen backbone.
- **Control 2b: slow-mode + stochastic group rescale.** 100 ps test using existing components; checks whether the temporal slow mode and the stochastic group rescale fix KE variance and D together.

Candidate 2 (fixed basis-induced rule) remains the integration candidate only after 2a/2b pass the KE/ACF/D gate.
