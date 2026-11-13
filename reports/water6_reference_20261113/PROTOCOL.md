# Stage A: Water on the actual project data

Authorized 2026-11-13 after `paper/NMI_RESEARCH_PLAN_20261113.md`. This stage
tests a mature direct propagator and the radius hypothesis, not a new method.
Historical failures, configurations and checkpoints remain intact.

## Frozen tasks and selection

64 flexible SPC/Fw water molecules (192 atoms), 5 fs prediction; the existing
5,000/1,250/1,000 transition split. Training sources train_01/02/03, validation
val_04, test test_05. No MD generation, relabeling, resplitting, rollout training,
physical loss or pretrained initialization. Read the test set for model
evaluation only after validation selection; graph preparation numerically
checks existing test labels without fitting or reporting a model score.

Both runs use the same rebuilt 6 Å graphs and original displacement and absolute
final-velocity targets. The pre-existing graph's type indices are H=0/O=1;
HDF5 LAMMPS type IDs are a different representation, checked through atomic
numbers rather than copied into the model. Positions may differ by integer
lattice images. Cached/online graph conventions are checked explicitly.

Select best by mean normalized position/velocity endpoint MSE on all 1,250
validation samples. Do not select on long-time physics or test results.

| Setting | DSRT radius-only | Native TrajCast |
|---|---|---|
| Config | `configs/water6_dsrt_20261113.json` | `configs/water6_trajcast_native_20261113.json` |
| Representation | unchanged R3-Opt, 192/48/96, 2+2 | official Water, l≤2, 64 channels, 4 layers |
| Precision | FP32, TF32 off | FP64 |
| Batch | 8 | 2 |
| Optimizer | AdamW AMSGrad, wd=0 | Adam AMSGrad, wd=0 |
| LR | 5k warmup 1e-5→3e-4, cosine→1e-5 | 0.01; plateau factor .8/patience 20 |
| Budget | 250,000 updates / 400 epochs | ≤1,500 epochs / 3,750,000 updates |
| Early stopping | no ordinary validation early stop | 100 stale validation epochs |
| Proposal auxiliary | historical .02 | absent in original model |
| Initialization | random, seed 20261103 | random, seed 20261103 |

These budgets and precisions differ deliberately. This is a same-data native
reference comparison, **not** a matched-compute architecture attribution.
Record actual updates, samples processed, wall time, GPU model, peak allocation,
shared utilization and environment. Shared-device throughput is not a clean
publication benchmark; remeasure final inference on an idle GPU.

## Native implementation boundary

Pinned IBM TrajCast commit `c52f107a02176671b21a420b2d4c67e0327b04f7`,
unmodified under `../upstream/trajcast-c52f107`; Apache-2.0 copyright and license
remain there. The published Water e3nn YAML is archived beside this document.
Velocity RBF maximum is 0.135 Å/fs; cutoff 6 Å; angular momentum constraint off;
native zero linear momentum setting retained. Do not claim energy conservation.

`water_reference.train_native` uses upstream `Trainer`, its training-only RMS
normalization, `EfficientTrajCastModel`, native momentum layer, MSE and `_train_epoch`.
The six-component MSE equals .5 times the sum of normalized q/v MSEs for equal
three-component outputs. Management additions are full validation, atomic
best/last including optimizer/scheduler/RNG, resume, progress and nonfinite
gradient rejection. No additional optimizer, network or loss is introduced.
Gradient clipping is left at the upstream default (infinity); the paper does
not specify a clipping threshold. Plateau uses PyTorch defaults except the
published factor/patience; this choice is recorded rather than claimed to be
an unavailable exact historical training configuration.

The pinned repository also contains an older `TrajCastModel` (1,993,000
parameters). It has a different initialization/first layer and does not match
the released Water state. The released Water model is `EfficientTrajCastModel`
(2,258,856 parameters); all state key names and tensor shapes were compared
against IBM's public e3nn state, without loading those parameters into training.
The initial wrong-entry launch was stopped and retained under
`rejected_wrong_native_class/`; it is excluded from model comparisons but its
updates/time remain part of setup expenditure. DSRT was not interrupted.

Official model config:
[IBM Water YAML](https://huggingface.co/ibm-research/trajcast.models-arxiv2025/blob/main/water/config_e3nn.yaml).
Training settings and citation: Thiemann et al., *Force-free molecular dynamics
through autoregressive equivariant networks*, NMI 8, 764–776 (2026),
[doi:10.1038/s42256-026-01227-7](https://doi.org/10.1038/s42256-026-01227-7).

## After training

First report held-out endpoint metrics and 1/2/4/8-step errors. Use the same
validation-source initial conditions for 1/5/20 ps NVE, recording full KE/PE/E,
temperature, molecular translation/rotation/internal KE and bond geometry.
Reference SPC/Fw is offline only. Preserve every failure. These checks decide
whether to spend time on complete 100 ps NVE / 2.5 ns NVT; they never select
another checkpoint. Native TrajCast inference must preserve its original
readout/normalization/conservation and Forecast update semantics; any adapter
must numerically agree before physical results are reported. Reuse the existing
CSVR, force field and RDF/VACF/VDOS/MSD definitions and disclose fit windows.

Do not launch a candidate l=2 DSRT, motion modes, new physical losses, on-policy
aggregation or a third training arm in this stage. New design follows the
actual-task comparison, rather than preceding it.
