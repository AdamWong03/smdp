# Method Change — Mode-Aware Collective/Internal Readout

**Date:** 2026-11-21
**Status:** implemented; 20k fine-tune running
**Target:** NMI method paper (locked)

---

## 1. Why this change

The NVT audit showed a **partition-direction** problem:

| Model | Translation | Rotation | Internal |
|---|---:|---:|---:|
| 250k Full | 0.538 | 0.786 | 6.044 |
| Current-Only | 0.646 | 0.535 | 6.384 |
| v0 | 0.111 | 0.808 | 6.338 |
| Native TrajCast ep145 | 3.256 | 3.105 | 1.041 |
| Reference | 2.373 | 2.430 | 2.609 |

Our models suppress collective motion and over-excite internal motion; native does the
opposite. The task is not forcing one failure direction. The representation/readout controls
the partition direction.

## 2. Mechanism

A frozen base propagator predicts the full per-atom velocity. A zero-initialized correction
adds two explicit paths:

- **collective**: per-molecule COM translation + angular velocity, predicted from pooled node
  scalars;
- **internal**: per-atom motion in the molecular frame, predicted from node scalars + vector
  norm.

```
v_final = v_base + scale * (dv_com[mol] + ω[mol] × r_rel + dv_internal)
dq_final = dq_base + h * correction
```

The correction is zero at initialization, so the fine-tuned model starts exactly at the stable
250k base. The base is frozen; only the correction heads are trained. This isolates whether the
mode-aware path itself can change the NVT partition.

## 3. Implementation

- `mode_aware.py`: `ModeAwareCorrection`, `ModeAwareTransition`.
- `finetune_mode_aware.py`: frozen-base fine-tune; endpoint loss only (proposal weight 0), best
  validation checkpoint + last checkpoint saved.
- `relational_inference.py`: loader supports `architecture='mode_aware_finetune'`.
- Zero-init equivalence check: wrapper output equals base exactly (dv=0, dq=0, corr=0).

## 4. Current run

| Item | Value |
|---|---|
| Run | `experiments/mode_aware_finetune_20k_20261121` |
| Base | 250k Full 6 Å (`dsrt_radius6/best.pt`) |
| Trainable | correction heads only |
| Steps | 20,000 |
| Batch | 8 |
| LR | 5e-5, warmup + cosine |
| GPU | L40S 1 |

Smoke at lr 1e-4/20 steps degraded endpoint; the 20k run uses 5e-5 and selects the best
validation endpoint, so it should not destroy the base. Both `best.pt` and `last.pt` will be
kept; NVT screen will evaluate both if they differ.

## 5. Evaluation

100 ps NVT screen (50 ps thermalization + 100 ps production), same val_04 start:

- translation/rotation/internal partition;
- RDF, VDOS;
- stability;
- endpoint RMSE.

**Success:** partition moves toward reference without instability or endpoint collapse.

**Falsifier:** no partition movement, or endpoint/instability degrades.

If the frozen-base correction works, the next step is an end-to-end from-scratch 250k run with
the mode-aware readout, then a clean NMI method claim. If it does not, the correction path is
not the lever; go back to the operator/representation rather than tuning this head.

---

## 6. Result (2026-11-21): null

20k fine-tune completed. The correction head learned almost nothing:

| Checkpoint | Correction RMS | Velocity RMS | Val position RMSE | Val velocity RMSE |
|---|---:|---:|---:|---:|
| Base 250k Full | 0 | 0.01242 | 0.000410 Å | 0.000150 Å/fs |
| Mode-aware best | 9.60e-6 | 0.01242 | 0.000412 Å | 0.000150 Å/fs |
| Mode-aware last | 5.72e-6 | 0.01242 | 0.000412 Å | 0.000150 Å/fs |

The correction is **0.08% of the velocity RMS**; endpoint is unchanged. A 100 ps NVT screen
would be identical to the base within noise, so it was not run.

**Why:** the base is already endpoint-optimal. Endpoint loss gives the correction no gradient
toward the physical partition; the correction can lower the loss only by staying near zero.
A mode-aware readout trained with endpoint supervision cannot fix the NVT partition.

**Conclusion:** the readout idea is not disproven, but it needs a **physical training signal**
(rollout/partition loss), or a from-scratch inductive bias. Endpoint-only fine-tuning is a
dead end.

## 7. Next method options

| Option | What it tests | Cost | Risk |
|---|---|---|---|
| **From-scratch mode-aware 250k, endpoint loss** | Does the readout inductive bias alone change partition? | ~13h | likely null; endpoint loss still misaligned |
| **Rollout/partition-aware fine-tune** | Can a physical rollout loss train the correction? | 2–4h | complex; memory/time; may be unstable |
| **Backbone/velocity coupling change** | Does the operator path control the bias? | ~13h | prior art; not novel alone |

Recommendation: **rollout/partition-aware fine-tune** is the only one that directly addresses
the root cause (endpoint loss does not constrain NVT partition). If it fails, the remaining
option is a backbone change, not another readout.

---

## 9. What the nulls imply

The failure is a **long-time/equilibrium** phenomenon:

| Signal | Sees the failure? |
|---|---|
| one-step endpoint/partition | no (almost perfect) |
| 2-step partition | no (loss ~1e-4, flat) |
| 20 ps NVE | misleading (v0 looked better, NVT failed) |
| 500 fs NVE rollout | no |
| 100 ps NVT | **yes** (bias visible in first 100 ps) |
| 2.5 ns NVT | yes |

Therefore a training signal must either:

1. **use states from a long autonomous NVT rollout** (on-policy correction), or
2. **use a structural prior** that constrains the partition without needing a long signal.

### Option A: on-policy correction (recommended, requires approval)

- Generate a 100 ps NVT trajectory with the base model.
- Sample autonomous states from it.
- At each sampled state, compute a one-step reference continuation with SPC/Fw
  (`diagnose_water_bias.py` already does this for diagnosis).
- Fine-tune the correction to match the reference one-step velocity/partition at those states.
- Inference remains force-free; reference calls are training labels only.

This changes the project's current rule that diagnostic labels are never used for training.
The research lead must explicitly approve it.

### Option B: long-horizon distribution loss / black-box optimization

- Generate long NVT rollouts (detached), optimize a small correction to match partition
  moments. No gradients through 20k steps; expensive, less principled.

### Option C: structural prior

- Separate collective/internal integration, constrained kinetic update, or a backbone change.
- Expensive and uncertain; native already shows a different bias.

**Recommendation:** Option A, because it is the only one that trains on the states where the
model actually fails. It is also a coherent method story: "on-policy correction against short
reference continuations fixes long-time equilibrium bias while keeping force-free inference."

---

## 8. Rollout/partition-aware fine-tune (running)

Implemented:

- `rollout_data.py`: samples 2-step sequences from raw NVE trajectories;
- `finetune_rollout_physical.py`: 2-step differentiable rollout with a mode partition loss at
  the final reference state;
- loss: `0.1*endpoint1 + endpoint2 + partition2`;
- base frozen; only the mode-aware correction head is trained;
- 256 train sequences / 32 val sequences; 5,000 steps; batch 4; lr 5e-5.

Smoke: 2 steps, partition loss 0.00986 on validation; endpoint2 0.0238. The correction can now
receive a physical gradient even when endpoint is already optimal.

Run: `experiments/rollout_physical_finetune_5k_20261121`.

Success: partition loss decreases on validation without endpoint/instability collapse; then
100 ps NVT screen. Falsifier: partition loss flat, or endpoint/instability degrades.

### Result: null

5k fine-tune completed. Validation partition loss was already ~1e-4 at epoch 1 and stayed there;
the correction RMS after training is 9.61e-6 (0.08% of velocity RMS), identical to the
endpoint-only fine-tune. Endpoint2 ~3.6e-4; no meaningful movement.

**Why:** the base model's **2-step** rollout partition is already close to reference. The NVT
failure appears over 100 ps; it is invisible at 2 steps. A short-horizon partition loss has no
gradient to teach the correction. This is the same lesson as one-step, 20 ps NVE and 500 fs
rollout, now at 2 steps.

**Conclusion:** short-horizon physical losses cannot fix the long-time NVT partition. The method
needs a long-horizon/equilibrium training signal, or a structural prior. Do not tune this head
further.

### Postscript 2026-11-21 — why the short-horizon null is fundamental

A one-step diagnostic on held-out NVE pairs showed that the base model's mode-resolved gain and
variance are already correct to ~1e-4, and the residual is only ~1e-4 of the mode variance.
The 2-step partition is correct for the same reason. The NVT failure is a **closed-loop
distribution shift**: a tiny state-dependent error moves the model off the training
distribution, where the error grows and pushes the partition further.

Therefore this is not a loss-weighting problem, and a one-step gain/spectral regularizer would
also see nothing. The root-cause signal is **on-policy**: states the model actually visits in a
long rollout, labelled with reference one-step continuations. See
`LONG_TIME_TRAINING_LITERATURE.md` §4 and GAP_LOG G14.

