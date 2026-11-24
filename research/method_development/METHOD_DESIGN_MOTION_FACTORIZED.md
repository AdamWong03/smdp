# Method Design — Motion-Factorized Direct Propagation

**Date:** 2026-11-20
**Status:** design proposal; no training launched
**Target:** NMI method paper (locked)
**Motivation:** the global velocity MSE is dominated by internal/rotational motion; translation
is underweighted. This is a concrete, testable cause of the kinetic-partition bias.

---

## 1. Diagnostic that motivates the design

On 200 Water training transitions, decompose the velocity MSE contribution into molecular
translation, rotation, and internal motion (same result for initial and final velocities):

| Component | Mean squared velocity contribution | Fraction of total |
|---|---:|---:|
| Translation | 1.213e-4 Å²/fs² | **7.8%** |
| Rotation | 6.78e-4 Å²/fs² | **43.8%** |
| Internal | 7.48e-4 Å²/fs² | **48.4%** |
| Total | 1.546e-3 Å²/fs² | 100% |

The global velocity loss therefore gives **translation ~12× less weight than internal/rotation**.
A model can lower the average loss by fitting internal/rotational motion and still be poor at
molecular translation — exactly the observed NVT failure (translation 0.152 vs 2.459 eV,
rotation 0.212 vs 2.502 eV, internal 7.330 vs 2.450 eV).

This is a hypothesis, not yet a result. It yields a concrete method change: **make the training
objective and the readout motion-aware.**

---

## 2. Core hypothesis

> The kinetic-partition failure is caused, at least in part, by a **loss/representation
> imbalance**: the direct propagator optimizes a globally normalized velocity MSE dominated by
> internal and rotational motion, so molecular translation is under-constrained. Factoring
> motion into collective (translation/rotation) and internal components, normalizing and
> weighting them explicitly, and training on short rollouts should preserve the partition
> without sacrificing endpoint accuracy.

If this is false, a mode-balanced loss will not change NVT partition; that is the falsifier.

---

## 3. Method components

### M-A. Mode-wise normalization and loss

For each molecule:

- translation velocity: COM velocity `v_com`;
- rotation velocity: `ω × r_rel` about the molecular COM;
- internal velocity: `v_rel - ω × r_rel`.

Normalize each component by its own TRAIN-only RMS, then use a weighted sum:

```
L_mode = w_t * MSE(Δq_t, Δq_t*) + w_v * [
    λ_trans * MSE(v_trans, v_trans*) / rms_trans^2
  + λ_rot   * MSE(v_rot,   v_rot*)   / rms_rot^2
  + λ_int   * MSE(v_int,   v_int*)   / rms_int^2
]
```

Initial weights: equal `λ_trans = λ_rot = λ_int = 1/3` (or tuned). This is a training change, no
new architecture. It directly tests the hypothesis.

### M-B. Mode-aware readout

Replace the single absolute-velocity head with:

- a collective head predicting molecular COM translation and rotation;
- an internal head predicting internal motion in the molecular frame;
- a recombination step that preserves the total velocity.

This is an architectural change, but a small one: the backbone stays; only the readout and
motion representation become mode-aware. It gives the method paper a mechanism beyond a loss
trick.

### M-C. Multi-horizon rollout training

Train on 2-step (and optionally 4-step) rollouts from raw NVE trajectories:

```
state_t -> model -> state_{t+1} -> rebuild graph -> model -> state_{t+2}
loss = L_mode(step 1) + L_mode(step 2)
```

This penalizes error accumulation and gives the mode-balanced loss a long-time signal. It uses
existing data; no new MD, no force/energy loss.

### M-D. Partition-aware selection

Use the H1 rollout partition proxy for checkpoint selection:

```
selection = endpoint_loss + w * rollout_partition_error
```

This is already implemented; it becomes part of the method rather than a standalone trick.

---

## 4. Staged experiments

| Stage | Change | Data/model | Budget | Success | Falsifier |
|---|---|---|---|---|---|
| **E1** | M-A mode-balanced loss only | Water 6 Å Full | 20k updates, 1 seed | translation improves in 20 ps NVE; endpoint not worse than baseline | no partition change |
| **E2** | M-B mode-aware readout | Water 6 Å | 20k, 1 seed | better translation/rotation without internal blow-up | no gain over E1 |
| **E3** | M-C 2-step rollout + M-A | Water 6 Å | 20k, 1 seed | lower 500 fs/1 ps rollout error; better NVE partition | no gain over E1 |
| **E4** | best of E1–E3 + M-D selection | Water 6 Å | 50k, 3 seeds | NVT partition/RDF/VDOS closer to reference | effect within seed variance |
| **E5** | Quartz boundary + cost | Quartz 30 fs | matched | Quartz preserved; cost measured | Quartz degrades |

Do not run E2–E3 until E1 shows a non-trivial partition movement. Do not run E4 until one of
E1–E3 beats the baseline at 20 ps NVE.

---

## 5. Why this is a method paper, not a trick

- **Mechanism:** the loss/representation is not measuring the observable we care about;
  motion factorization changes what the model is trained and represented to preserve.
- **Representation:** mode-aware readout separates collective and internal dynamics.
- **Training:** mode-balanced multi-horizon objective.
- **Evaluation:** NVT partition/RDF/VDOS/diffusion at matched cost.
- **Delta:** the closest works (TrajCast/ChronoCast/FlashMD) do not factor molecular motion
  into collective/internal channels for a direct propagator; FlashMD uses an energy criterion,
  not a kinetic-partition criterion.

If only the loss changes and the gain is small, the paper is a training-method paper, not an
NMI method paper. The architectural component M-B is what makes it a method contribution.

---

## 6. Prior art / delta

| Prior work | What it does | What it does not do |
|---|---|---|
| TrajCast | velocity tensor product; global Δr/v MSE | no molecular mode factorization; no partition selection |
| ChronoCast | velocity node feature; momentum conservation | no molecular modes; crystal only |
| FlashMD | momenta + Hamiltonian structure; energy-based selection | no molecular translation/rotation/internal factorization |
| EGNO | velocity directional tensor; trajectory MSE | no liquid water; no partition |
| Classical MD | internal coordinates / mode decomposition | not a learned direct propagator |

The delta is: **motion-factorized representation + mode-balanced multi-horizon training +
partition-aware selection for direct finite-time propagators.** It is specific, testable, and
not just "more l=2."

---

## 7. Risks

1. **Translation-only gain.** Mode balancing may improve translation but not rotation/internal;
   that is still useful but not a full fix.
2. **Weight sensitivity.** Equal weights may overcorrect; tune on validation, not test.
3. **Water-specific.** The decomposition uses molecular topology; quartz has no molecules. The
   method is for molecular liquids; quartz is a boundary.
4. **NMI breadth.** One molecular liquid may be too narrow; a second molecular liquid would
   strengthen the paper but requires new data.
5. **Cost.** Mode-aware readout adds parameters/ops; measure against the 6 Å baseline.
6. **Prior-art surprise.** Molecular-mode ideas exist in classical MD and coarse-graining; the
   novelty is the direct-propagator application, not the decomposition itself.

---

## 8. What not to do

- Do not launch E2–E5 before E1 shows a signal.
- Do not change architecture and loss at the same time in the first experiment.
- Do not claim NMI readiness until NVT + 3 seeds + cost + Quartz boundary are complete.
- Do not treat a loss-only gain as a method paper; it is a component.

---

## 9. E1 implementation status (2026-11-20)

Implemented in `code/h1h2` on branch `research/h1h2-selection-20261120`:

- `motion_loss.py` — differentiable mode decomposition, TRAIN-only mode RMS, motion-balanced
  objective;
- `train_relational.py` — loss branch, mode metrics in `status.json`, endpoint selection
  preserved for the E1 isolation;
- `tests/test_motion_loss.py` — 5 tests pass (pure translation/rotation/internal,
  differentiability, RMS/MSE);
- configs: `water6_e1_baseline_20k_20261120.json` and
  `water6_e1_motion_20k_20261120.json`;
- CPU integration check: forward + loss + backward on a real `PhaseSpaceTransition` batch,
  finite gradients.

Launch plan: the E1 smoke runs first (20 steps, batch 2); if it completes, the 20k A/B runs
on L40S 1/4. The baseline and motion arms use the **same endpoint selection** so the only
difference is the training loss.

**Status 2026-11-20:** E1 completed; **negative as implemented**. Baseline 20k: translation
2.421, rotation 1.820, internal 2.873, total 7.114 eV; motion: 2.689, 2.081, 5.396, 10.166 eV.
The motion arm heats to ~400–420 K and is worse on RDF/VDOS. Likely causes: the mode-balanced
objective was also applied to the proposal (proposal loss 0.387 vs 0.017), and equal mode
weights overcorrected translation. See `E1_MOTION_LOSS_RESULT.md`.

**Unexpected signal:** the 20k baseline has worse endpoint but better 20 ps NVE partition than
the 250k Full 6 Å model (internal 2.873 vs 4.418 eV; total 7.114 vs 7.321 eV; H/O VDOS
0.850/0.914 vs 0.717/0.884). This is not a controlled seed comparison, but it suggests that
longer endpoint training may degrade physical fidelity. The higher-priority next experiment is
a **controlled training-time sweep** (E0), not scaling E1.
