# Geometric-learning control report

Date: 2026-11-27  
Status: **bounded diagnostics completed; no long MD, no full training**

This report tests whether the ParityFlow V1 even-head collapse is a data,
target, normalization, or readout problem, or whether it is a defect of the
ParityFlow V1 architecture/optimization path. The parity decomposition itself
is not put on trial.

## 1. Fixed states and units

All controls use the same centered states:

- Water 5 fs: 64 TRAIN and 64 held-out TEST states, seed 20261127.
- Quartz 30 fs: 64 TRAIN and 64 held-out TEST states, seed 20261127.
- TRAIN-only normalization; no validation/test fitting.
- Physical units: Å, fs, u. Velocities and displacements are mass-centered
  graph by graph; positions remain absolute.
- Targets are the genuine t−h/t/t+h states. Water and quartz targets were
  independently recomputed from the raw NVE frames and match to float32
  precision. Mass-weighted COM is ~1e-7. The quartz index drops one sample with
  no t−h frame; index row 0 maps to original sample 1 via `sample_index`.

The 64 TRAIN states are the same subset used in the ParityFlow V1
TRAIN-only overfit diagnostics (`torch.randperm(..., seed=20261127)[:64]`).

## 2. Odd residuals: correcting the parity interpretation

The original odd metrics are dominated by the ballistic/identity term. We
therefore report the interaction-dependent residuals in physical absolute
units:

```
Q_odd_residual = Q_odd - h * p/m
P_odd_residual = P_odd - p
```

| model | system | Q_odd MSE/zero | Q_res MSE/zero | Q_res R² | Q_res corr | P_odd MSE/zero | P_res MSE/zero | P_res R² | P_res corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ParityFlow V1 | water 5 fs (4k) | 0.0564 | 0.0918 | 0.9082 | 0.9531 | 0.0715 | 0.2110 | 0.7890 | 0.8883 |
| ParityFlow V1 | quartz 30 fs (16k) | 0.0273 | 0.0079 | 0.9921 | 0.9961 | 0.0888 | 0.0301 | 0.9699 | 0.9848 |

Even components on the same states:

| model | system | Q_even MSE/zero | Q_even R² | P_even MSE/zero | P_even R² |
|---|---|---:|---:|---:|---:|
| ParityFlow V1 | water 5 fs | 0.9993 | 0.0007 | 0.9992 | 0.0008 |
| ParityFlow V1 | quartz 30 fs | 0.8725 | 0.1275 | 0.9595 | 0.0405 |

Target/prediction RMS (same physical units) makes the even collapse explicit:

| model | system | Q_even target/pred | P_even target/pred | Q_odd target/pred | P_odd target/pred |
|---|---|---:|---:|---:|---:|
| ParityFlow V1 | water | 0.02752 / 0.00175 | 0.01144 / 0.00081 | 0.04782 / 0.04754 | 0.03477 / 0.03360 |
| ParityFlow V1 | quartz | 0.05152 / 0.01842 | 0.04772 / 0.00966 | 0.04934 / 0.04792 | 0.05126 / 0.04881 |

**Observation.** The odd heads do learn interaction-dependent residuals:
water Q/P residual R² = 0.91/0.79; quartz = 0.992/0.970. The original odd
metrics overstate this because the ballistic term dominates the target
variance, but the residuals are not zero. The even heads remain at the zero
predictor, with only a weak quartz Q_even correlation.

Full data: `odd_residual_diagnostics.json`; per-species breakdown included.

## 3. Official TrajCast positive control

Architecture: official `EfficientTrajCastModel`, 2,258,856 parameters,
64 hidden channels, 4 message-passing layers, lmax=2, 8 edge RBF, 8 velocity
RBF, e3nn backend, FP64, random initialization.

Training on the 64 TRAIN states, both physical successors:

```
(q,  v) -> (q_plus,  v_plus)
(q, -v) -> (q_minus, -v_minus)
```

Primary settings: Adam, lr 0.01, AMSGrad, weight decay 0, batch 2, no gradient
clipping, 2,000 updates. Water cutoff 6.0 Å; quartz cutoff 5.0 Å, `vel_max`
computed from TRAIN (0.02326 Å/fs), `avg_num_neighbors` computed from the fixed
graphs (89.83 water, 41.14 quartz).

| control | system | Q_even MSE/zero | Q_even R² | Q_even corr | P_even MSE/zero | P_even R² | P_even corr |
|---|---|---:|---:|---:|---:|---:|---:|
| TrajCast random 2k | water | 0.0384 | 0.9616 | 0.9809 | 1.021 | -0.021 | 0.1115 |
| TrajCast random 2k | quartz | 0.0316 | 0.9684 | 0.9841 | 1.613 | -0.613 | 0.0274 |
| TrajCast official pretrained | water | 0.0647 | 0.9353 | 0.9741 | 0.1068 | 0.8932 | 0.9529 |

Odd/residual metrics for the same controls:

| control | system | Q_odd MSE/zero | Q_res R² | P_odd MSE/zero | P_res R² |
|---|---|---:|---:|---:|---:|
| TrajCast random 2k | water | 0.0039 | 0.9936 | 0.9185 | -1.6957 |
| TrajCast random 2k | quartz | 0.0204 | 0.9941 | 1.4400 | 0.4875 |
| TrajCast pretrained | water | 0.0033 | 0.9947 | 0.0120 | 0.9931 |

**Observation.** The official pretrained TrajCast checkpoint, evaluated on the
same 64 held-out states and target convention, learns both even components:
Q_even R²=0.935, P_even R²=0.893. The from-scratch 2k official runs learn the
displacement components strongly (Q_even R² rises from ~0 at update 0 to
0.96–0.97 at 2k; the curves are in `controls/trajcast_*_2k.jsonl`) but the
velocity head remains at zero, so P_even/P_odd fail and the P_odd residual is
worse than the original P_odd. The from-scratch 2k run is therefore **not** a
clean positive control for the velocity/even-momentum response; it is a
budget/optimization result. The pretrained checkpoint is the clean
central-state positive control.

The optional ParityFlow-optimizer/precision arm (AdamW AMSGrad, lr 2e-4, batch
4, FP32, 1,000 updates) was worse: Q_even R²≈0, Q_odd R²=0.552, P_even R²≈0.
This does not rescue the velocity head and is not a preferred setting.

Paired endpoint/parity equivalence was verified numerically (displacement
absolute differences ~1e-19–1e-10; velocity differences at float precision).
Both physical target pairs were evaluated independently; learning curves and
per-step metrics are in `controls/trajcast_*_2k.jsonl`.

## 4. Geometry-only PaiNN control

Architecture: PaiNN-style scalar-vector encoder, 64 scalar / 32 vector
channels, 3 layers, 16 RBF channels, 231,168 parameters. Inputs: local periodic
geometry, species, masses. **No global Cartesian position, no momentum.**
Outputs: Q_even and P_even directly. TRAIN-only normalization; AdamW AMSGrad,
lr 2e-4, batch 8, 5k updates water / 2k quartz.

| system | updates | Q_even MSE/zero | Q_even R² | Q_even corr | P_even MSE/zero | P_even R² | P_even corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| water | 5,000 | 0.3517 | 0.6483 | 0.8194 | 0.5305 | 0.4695 | 0.6865 |
| quartz | 2,000 | 0.3472 | 0.6528 | 0.8121 | 0.8022 | 0.1978 | 0.4471 |

**Observation.** Local geometry alone contains substantial information about
Q_even (≈65% of variance on both systems) and a smaller but nonzero part of
P_even (47% water, 20% quartz). The water Q_even test R² rises from 0.06 at
update 400 to 0.65 at 5k; P_even rises from 0.22 to 0.47. Quartz reaches a
similar Q_even plateau by 2k but P_even remains at 0.20. P_even has an
irreducible momentum-dependent component for a geometry-only model, as
anticipated; this is stronger for quartz.

## 5. Existing trained propagators

These are frozen checkpoints evaluated on the same 64 held-out states, with
paired p/−p inputs and the same parity reconstruction. They are **not** a
matched training comparison: R3-Opt and Phase-Space V0 use future-conditioned
relations, and their cutoffs/horizons differ. They test target learnability and
the parity reconstruction, not an architecture ranking.

| model | system | Q_even MSE/zero | Q_even R² | P_even MSE/zero | P_even R² | Q_odd MSE/zero | P_odd MSE/zero |
|---|---|---:|---:|---:|---:|---:|---:|
| R3-Opt | water 5 fs | 0.00050 | 0.9995 | 0.00233 | 0.9977 | 2.0e-5 | 7.4e-5 |
| Phase-Space V0 | water 5 fs | 0.00023 | 0.9998 | 0.00237 | 0.9976 | 2.8e-5 | 1.5e-4 |
| R3-Opt | quartz 30 fs | 0.00092 | 0.9991 | 0.00178 | 0.9982 | 3.3e-4 | 7.4e-4 |
| Phase-Space V0 | quartz 30 fs | 0.00100 | 0.9990 | 0.00197 | 0.9980 | 3.8e-4 | 8.9e-4 |

Odd residuals are also recovered (MSE/zero ≈1e-4–1e-3; R²≈0.9997–0.9999).
Per-species errors, error covariance, and neighbor-count bins are in
`existing_propagators_results.json`.

**Observation.** At least two independently trained propagators recover the
even components almost perfectly on the same held-out targets. This rules out
the targets, normalization, and graph construction as the cause of the
ParityFlow even collapse. R3/Phase-Space use future conditioning, so they do
not establish a matched central-state architecture result; the pretrained
TrajCast checkpoint does provide that central-state positive control.

## 6. Interpretation

**What is established by observation**

1. The odd residuals contain genuine interaction learning, but the original
   odd MSE/zero overstates it. Corrected residual R² is 0.79–0.99.
2. ParityFlow V1's even components are at the zero predictor on both systems;
   the error is not a target-construction, COM, normalization, or
   tensor-product-selection bug.
3. The official pretrained TrajCast central-state model learns both even
   components (Q_even R²=0.935, P_even R²=0.893) on the same held-out states.
4. Fully trained R3-Opt and Phase-Space V0 recover even and odd components at
   R²≈0.998–0.9999 on the same states (with future conditioning).
5. A geometry-only PaiNN learns Q_even R²≈0.65 on both systems and P_even
   R²≈0.47/0.20, without momentum or global position.
6. The 2k random-init official TrajCast learns displacement but leaves the
   velocity head at zero; the pretrained full model learns velocities. This is
   a budget/optimization effect, not a target or harness defect.

**Hypotheses, not established causal layer claims**

- ParityFlow V1 is defective relative to successful central-state propagators
  in its even/force representation or optimization path. The audit found the
  even l=1 stream dominated by a global q feature and large node-local
  products, with the direct geometric message only ~2–3% of the block. This is
  consistent with the collapse but is not a causal layer identification.
- Local geometry is sufficient for a substantial part of Q_even, but P_even
  has an irreducible momentum-dependent component, especially for quartz. A
  geometry-only model cannot be expected to explain all of P_even.
- The 2k random-init TrajCast velocity failure is likely due to the short
  budget (62 epochs) / initialization of the velocity head, not the
  architecture; the pretrained model and the existing propagators prove the
  targets are learnable.

**What is not claimed**

- No claim that a specific ParityFlow layer is causally responsible.
- No claim that the parity decomposition is falsified.
- No claim of a matched architecture ranking between R3-Opt, Phase-Space V0,
  TrajCast, and PaiNN; their training data, conditioning, cutoffs, and
  objectives differ.
- No claim that the 2k random-init TrajCast run is a full external baseline.

## 7. Decision and next step

The positive controls succeed: a central-state official TrajCast checkpoint and
two trained propagators learn the even components that ParityFlow V1 does not.
The geometry-only control shows local geometry carries substantial Q_even
information, with P_even only partially reducible. Per the stopping rules, the
next methodological decision should treat ParityFlow V1 as defective relative
to a successful central-state propagator, not as evidence against the parity
decomposition.

No long-duration MD and no full-scale training were launched. The next round
should decide whether to redesign the even/force representation or to pursue a
different central-state architecture; this report does not authorize either.

## 8. Provenance and reproduction

Environment:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH=code/parityflow:code/h1h2:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib
```

Checkpoints:

- ParityFlow V1: `paused_v1/water5/last.pt` (4,000), `paused_v1/quartz30/last.pt` (16,000).
- Official TrajCast: `upstream/trajcast-official-water-reference/state_dict_e3nn.pt`
  (IBM `ibm-research/trajcast.models-arxiv2025`, 2,258,856 parameters).
- TrajCast commit: `c52f107a02176671b21a420b2d4c67e0327b04f7`.
- Official config: `water/config_e3nn.yaml` from the official model repository.
- R3-Opt water: `/data1/.../relational_transition/water5_20261103_r3_opt/best.pt`
  (epoch 399).
- R3-Opt quartz: `/data1/.../relational_transition/quartz30_20261103_r3_opt/best.pt`
  (epoch 363).
- Phase-Space water: `/data1/.../phase_space_20261120/water5/best.pt` (epoch 397).
- Phase-Space quartz: `/data1/.../phase_space_20261120/quartz30/best.pt` (epoch 395).

Key commands:

```bash
python code/parityflow/odd_residual_diagnostics.py \
  --out experiments/parityflow_20261127/odd_residual_diagnostics.json

python code/parityflow/positive_control_trajcast.py \
  --system water --index experiments/parityflow_20261127/indices/water5 \
  --updates 2000 --eval-every 200 \
  --out experiments/parityflow_20261127/controls/trajcast_water_2k.json

python code/parityflow/positive_control_trajcast.py \
  --system quartz --index experiments/parityflow_20261127/indices/quartz30 \
  --updates 2000 --eval-every 200 \
  --out experiments/parityflow_20261127/controls/trajcast_quartz_2k.json

python code/parityflow/geometry_control_painn.py \
  --system water --index experiments/parityflow_20261127/indices/water5 \
  --updates 5000 --out experiments/parityflow_20261127/controls/painn_water_5k.json

python code/parityflow/geometry_control_painn.py \
  --system quartz --index experiments/parityflow_20261127/indices/quartz30 \
  --updates 2000 --out experiments/parityflow_20261127/controls/painn_quartz_2k.json

python code/parityflow/evaluate_existing.py \
  --out experiments/parityflow_20261127/existing_propagators_results.json

python code/parityflow/finalize_controls.py
```

Artifacts:

- `positive_control_results.json`
- `odd_residual_diagnostics.json`
- `controls/` JSON and JSONL curves
- `existing_propagators_results.json`
- `CHECKPOINT_MANIFEST.json` (SHA-256 and byte sizes; weights remain under
  `experiments/parityflow_20261127/controls/`)
- `code_snapshot/` and repository snapshot
