# R3: Dynamics-consistent learned motion modes

**Status: implementation committed, training NOT run.**
Base commit: R2 reanalysis `7db6313`. Branch:
`experiment/r3-dynamics-consistent-modes`. All R1/R2 checkpoints,
trajectories, BKS physics definitions, original data splits and historical
results remain unchanged.

## Scientific question

Can independently learned atomic motion modes, heterogeneous propagation
and coupled latent evolution jointly produce an accurate and efficient
finite-time molecular-dynamics flow map? The central claim is **not**
conservation-by-construction, equal expert loads, or discovery of phonon
labels. NVE energy, NVT VDOS, Si–O/O–O RDF, and wall-clock throughput remain
independent tests of the quality of the learned dynamics.

The previous R2 showed improvements in some NVT statistics and output mode
usage but 0–5 ps NVE energy loss, later energy injection, worse endpoint RMSE
and slower dense-output runtime. Neither enforcing energy only nor extending
pushforward to more steps is justified as the automatic next architecture.

## What was implemented

**R3-A** (control): identical R2 topology including four motion tokens,
recurrent/direct propagators, atom-local cross-mode exchange, additive
readout and time-query decoder. A new NVE-only **flow composition** objective
trains the 16 fs direct state map to agree with two autonomous 8 fs maps:

```
F16(z) = NeuralSMDP(z, times_fs=[16])
z8     = NeuralSMDP(z, times_fs=[8])
F8+8(z)= NeuralSMDP(z8, times_fs=[8])
Lcomp  = MSE((F16.q - F8+8.q)/train_q_scale)
       + MSE((F16.v - F8+8.v)/train_v_scale)
```

`F8+8` rebuilds the periodic graph from the **predicted** 8 fs state.
Both sides are differentiable: this is not a detach-teacher distillation.
The existing reference-target position/velocity loss still directly
anchors 8 fs and 16 fs to **real NVE** samples; composition alone admits
incorrect fixed points. Composition is NVE-only because CSVR NVT is stochastic
and an independently sampled thermostat noise history cannot be composed
without accounting for the noise. No BKS energy or reference force call
enters training or learned propagation.

The original multi-time forward already contains 8 and 16 fs predictions,
so R3-A needs only one additional 8 fs inference per selected training
sample. Default in the R3 experiment: coefficient `0.05`, first two
samples per shuffled mini-batch; in fixed-order validation use the same
first two. This deliberately bounds overhead. Composition is measured
separately, not disguised as a lower supervised endpoint RMSE.

**R3-B** differs from R3-A in *one* feature: one sparse equivariant
`SpatialModeExchange` pass between the pre-existing atom-local mode
interaction and the second propagation stage. It communicates per-mode
scalar/vector state to neighboring atoms on the segment's start-of-step
PBC graph. The preceding atom-local cross-mode interaction allows
information to move along both atom and mode axes without evaluating
all K×K mode pairs on every edge. Coefficients depend only on scalar
invariants and radial functions; vector messages combine transformed
equivariant source channels and relative edge directions with invariant
coefficients. Envelope/degree normalization handles cutoff smoothly.
Graph rebuilt every **autonomous macrostep**, not at each latent half-step;
there is no repeated full equivariant encoder. This is a lightweight test of
propagation-time *spatial* coupling, not an assertion that the reference
trajectory's time-varying neighbor graph has been reconstructed.

`spatial_mode_exchange=False` remains backward-compatible with R2.
The optional spatial module is created **after** all shared parameters,
preserving the shared initialization under a matched seed. Subsequent
training RNG is explicitly reset for both R3 arms.

## Training and comparison

Same independent Quartz source trajectories and 5000/1250/1000 split.
No new reference trajectories. Use NVE segment + 4-step then 8-step
pushforward training. Do not automatically add a 16-step training phase.
Validation for all rollout phases uses the **same 16-step** loss and
same 1245 long-window validation starts. R3 selects the best checkpoint
over BOTH rollout phases, rather than forcing the last phase's first epoch.
No test or 100 ps NVE is used for checkpoint selection. Original R2
training remains available with default `--r3-composition-weight=0`,
`--max-rollout-steps=16` and no spatial messages.

The implementation intentionally does **not** rework R2's truncated-gradient
pushforward loss. That factor is held fixed across A/B so this experiment
tests *temporal composition* and then *spatial coupling*. Claiming effects
of corrected truncation would require a separate controlled experiment.

The paired run entrypoint is:

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
git switch experiment/r3-dynamics-consistent-modes
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
PYTHON=/data1/wyd/envs/mlip/bin/python bash scripts/run_r3_pair.sh
```

This script runs tests, trains both models **from scratch**, evaluates
both on common validation windows and times endpoint-vs-dense inference.
It DOES NOT generate new NVE/NVT trajectories or use test coordinates.
Output directories are separate and protected by existing no-overwrite
checks. Configure `PYTHON`, `ROOT`, `DATA`, `EXPERIMENTS` as needed.

For one arm only:

```bash
/data1/wyd/envs/mlip/bin/python train_quartz.py \
 --data-root /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1 \
 --output /data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3a_composition \
 --r3-composition-weight 0.05 --r3-composition-samples 2 \
 --max-rollout-steps 8 --seed 20261022
```

R3-B uses the same command in a different directory plus
`--spatial-mode-exchange`.

After *validation-stage* assessment, run any selected model through the
existing unchanged test/NVE/NVT protocol, retaining both 16 fs endpoint
and 1 fs dense-query comparisons. Both physical fidelity and real runtime
must be reported; there is no predicted performance claim.

## Validation criteria and integrity

- Unit-level: identity at t=0, gradients through 8+8 path, rotation and
  translation equivariance, periodic image consistency, mass-weighted COM
  conservation, empty graph behavior, and exact shared-weight initialization.
- Scientific diagnostics: endpoint and 16-step position/velocity accuracy,
  measured 8+8-versus-16 discrepancy, NVE total energy *time series*
  (not just final minus initial), NVT VDOS plus absolute kinetic statistics,
  Si–O and O–O RDF, and independent test trajectories only after selection.
- Runtime: measure separately **one 16 fs endpoint query** and all 16
  one-fs queries; the latter is useful for VDOS but costs more. The benchmark
  uses the same fixed input states for both variants and excludes thermostat
  and trajectory I/O, so do not conflate its ns/day with production throughput.
- Failure criterion: if R3-A lowers composition discrepancy but not physical
  fidelity or damages endpoint accuracy, consistency is not a validated
  improvement. If R3-B merely increases complexity/latency without a
  measurable benefit to trajectory fidelity or learned mode interaction,
  reject the added spatial block. Nothing is kept because its label sounds
  theoretically appealing.

## Prior work used for the implementation

- Boffi, Albergo, Vanden-Eijnden, *How to build a consistency model:
  Learning flow maps via self-distillation*, NeurIPS 2025:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/2ff3605c9e442d55156d99fe4f5f228d-Abstract-Conference.html
  (flow-map composition is not novel *per se*; our question is what it does
  for long-stride coupled atomic motion modes).
- Wu et al., *Equivariant Spatio-Temporal Attentive Graph Networks for
  Physical Dynamics Simulation*, NeurIPS 2023:
  https://github.com/GLAD-RUC/ESTAG
  (spatiotemporal coupling and the need to preserve equivariance).
- Fu et al., *Learning to Simulate Time-integrated Coarse-grained Molecular
  Dynamics with Multi-scale Graph Networks*, TMLR 2023:
  https://openreview.net/forum?id=y8RZoPjEUl
  (MD context and relevance of multiscale graph propagation).

**Do not interpret implementation and unit tests as a completed experimental
result.** Frozen experiment metrics must be filled only after server runs.
