# Relational endpoint network — discussion prototype

Implementation: `relational_transition.py`. This is an independent model,
not a subclass of TrajCast and not a modification of historical checkpoints.
Formal endpoint training was authorized and launched on 2026-11-02; see the
run section below. No physical-performance claim is established yet.

The architectural question is whether an explicitly supervised candidate
endpoint helps subsequent relational computation correct a joint prediction.
Prediction/refinement and coordinate-updating GNNs have precedents; this is
not a claim of an unprecedented architecture or established paper contribution.

## Forward computation

1. Encode elements, masses, per-system COM-relative velocities and periodic
   edge geometry into scalar/vector node states and persistent scalar edge states.
2. Two relational blocks update edge and node features jointly.
3. An equivariant head proposes displacement and final velocity at **30 fs**.
4. Recompute relative positions/velocities on the SAME periodic edges using
   this proposal. Preserve original relations alongside candidate relations.
5. Two further relational blocks consume both relation sets and carried node/
   edge states. A final equivariant head corrects the SAME 30 fs endpoint.

The intermediate proposal is not a 15 fs state. There are no internal physical
integration steps, force calls, mode tokens, action functions or implicit solvers.
Candidate geometry remains differentiable, including its dependence on the
proposal head. Final loss trains both stages through this path.

Default dimensions: 128 scalar / 32 polar-vector / 64 edge channels, 16 radial
basis functions, cutoff 4.5 A, 2+2 blocks, **853,920 trainable parameters**.
All vector nonlinearities are invariant scalar gating of equivariant vectors;
channel linear maps have no xyz bias. Each physical graph independently retains
its input total momentum and COM drift. This does NOT imply energy conservation.

## Existing data interface

`model(AtomicGraph_or_Batch)` returns a dictionary with `displacements`,
`positions`, `velocities`, proposal outputs, corrections, and normalized `target`.
It accepts existing official graph caches without changing or regenerating MD.
AtomicGraph edges are `[central, neighbor]`; the implementation converts them
internally to sender/receiver convention. Cartesian periodic shifts follow the
official `pos[edge[1]] - pos[edge[0]] + shifts` formula.

`endpoint_objective` uses real displacement/final-velocity labels, separate
normalization scales, and auxiliary proposal weight 0.1. That weight is a
provisional implementation default, not a tuned result. For actual training,
pass RMS scales computed on the training split; constructor scale defaults are
only numerical interface defaults, not dataset statistics. No interior labels
are used. A timestep inconsistent with the configured fixed horizon is rejected.

Use the project's existing per-system neighbor reconstruction on every new
autoregressive state. Candidate refinement retains initial edges; support
coverage for large rearrangements is an unverified limitation. No integration
with the old segment-loss trainer is implied; `train_relational.py` provides
endpoint training on the existing graph cache instead.

## Training-compute implementation

The first edge affine map is evaluated exactly as
`W_i s_i + W_j s_j + W_e e`: node projections are computed per node and gathered,
rather than recomputed per edge from a large concatenated tensor. Vector sender
projection is also evaluated per node. No parameters or nonlinearities are
removed. Eight correctness checks include output AND gradient agreement with
the unfactored implementation.

FP32 forward/backward is checked. BF16 autocast can be measured by the benchmark
but is NOT yet approved as the training precision. Reductions retain state
precision. No speedup over old SMDP is asserted from parameter count.

Commands from repository root:

```bash
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
OMP_NUM_THREADS=2 "$PY" -m pytest tests/test_relational_transition.py -q

# Explicit device selection; run only when resources are available.
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 "$PY" -m scripts.benchmark_relational_transition \
  --cache /data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028/cache \
  --device cuda --precision fp32 --batch-sizes 2 8 16 --iterations 30 \
  --output reports/relational_prototype/gpu_throughput.json
```

Benchmark includes forward, endpoint objective, backward and AdamW at zero
learning rate. No weights are fitted/saved. It uses training graphs only and
resident data, so its throughput excludes the data loader and validation.

Correctness checks cover O(3) including reflection, translations, uniform
velocity boosts, permutation, independent batch COM, periodic reimaging with
neighbor rebuilding, gradients through candidate relations, feedback influence,
affine factoring and FP32 gradients. A separate actual-data check executed
forward/backward on two cached Quartz training samples (324 atoms, 9,320 edges)
with finite gradients for every trainable module. None of these checks establishes
accuracy, stability, or a hardware training-time estimate.

## Source-level comparison informing the prototype

These are code-reading findings, not claims that the implementations have all
been integrated. Upstream files were read without installing their dependencies
or modifying the existing environment.

- TrajCast: existing local pinned implementation supplies periodic graph/data
  conventions. Its entire network is not inherited by this prototype.
- GNS: `learning_to_simulate/graph_network.py`, `EncodeProcessDecode._process_step`
  updates both node and edge latent states with residuals. This is a precedent
  for persistent relational processing, not our novelty. Its TensorFlow/Sonnet
  simulator is not a drop-in O(3)-equivariant PyTorch module.
  https://github.com/google-deepmind/deepmind-research/blob/master/learning_to_simulate/graph_network.py
- MDtrajNet: the actual equivariant model is in `mlatom/EEEDDDD.py`, particularly
  `E3AttnBlk.forward` and `EEENNDDDD.forward`. Each block updates species,
  coordinates, velocity features and time features; the next block recomputes
  geometry from the updated coordinates. The initial neighbor mask/indices are
  reused. Final public prediction returns coordinates; physical velocities can
  be computed by time differentiation in the wrapper. Consequently, geometric
  prediction feedback itself is NOT a new mechanism. `EEEDDDD_nn.py` contains
  generic MLP/attention helpers and should not be cited as the complete model.
  https://github.com/dralgroup/mlatom/blob/main/mlatom/EEEDDDD.py
- GotenNet: `gotennet/models/representation/gotennet.py`, `GATA.forward/message`
  computes node projections before edge propagation, combines invariant
  attention/spatial filtering with steerable features, and `edge_update`
  contracts projected directional features to modulate persistent edges.
  Its geometric contractions are a useful reference for richer relation
  descriptors, but adding them has not yet been justified for this prototype.
  https://github.com/sarpaykent/GotenNet/blob/main/gotennet/models/representation/gotennet.py
- E2Former-V2: inspect the **e2formerv2 branch**, not the original E2Former.
  `E2AttentionArbOrder_sparse.forward` in `src/molfm/models/e2former/e2former.py`
  projects queries/keys/values on nodes and invokes indexed Triton QK/value
  aggregation. The value kernel expects node values, padded neighbor indices
  and attention coefficients with specific layouts; it is not equivalent to
  this prototype's nonlinear edge MLP and cannot simply replace index_add.
  The inspected value kernel supports float16/bfloat16/float32, not float64.
  https://github.com/IQuestLab/UBio-MolFM/tree/e2formerv2/src/molfm/models/e2former

The current candidate distinction is explicit supervised *physical endpoint*
q/v followed by joint original/candidate relation conditioning, not merely
updating coordinates between GNN layers. Its empirical value and novelty remain
unestablished. No attention, extra expert or fused custom kernel was added just
because it appears in a reference implementation.

CPU-only measured affine-factorization result (`reports/relational_prototype/
cpu_throughput.json`, 2 threads, FP32, existing training graphs, 7 measured steps):

| Batch | Unfactored step | Factored step | Ratio |
|---|---:|---:|---:|
| 2 | 0.5649 s | 0.3380 s | 1.67x |
| 8 | 2.2747 s | 1.4747 s | 1.54x |

These are resident forward/backward/zero-lr-AdamW measurements on a shared CPU,
not GPU speedups, training convergence results or ns/day estimates. GPU devices
were occupied when checked; no existing GPU jobs were interrupted.

## Formal training authorized 2026-11-02

The user subsequently requested stopping the old segment/rollout training and
training this model. Old service `smdp-quartz30-20261030-gpu1` was stopped and its
best/last weights and physical results retained. GPU 1 (L40S) is now used by
`smdp-relational-quartz30-20261102.service`; other jobs are untouched.

Run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r1`.
Config/metadata copies: `reports/relational_quartz30_20261102/`.
Uses all 5,000 train / 1,250 validation / 1,000 test endpoints, fixed seed
20261102, batch 8, FP32 without TF32, AdamW with zero weight decay and AMSGrad,
LR 0.001, plateau scheduler factor 0.8/patience 20, maximum 600 epochs,
early stopping patience 80, gradient clipping 10, auxiliary proposal weight 0.1.
Model selection uses final endpoint validation loss; test coordinates are loaded
only after training and selection finish. Normalization is recomputed from train.
Only best/last checkpoints are saved atomically, including optimizer, scheduler
and RNG state. SIGTERM requests a stop at the next complete epoch boundary.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /data1/wyd/envs/trajcast-c52f107/bin/python -m train_relational \
  --run /data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r1 \
  --resume
```

Initial measured complete epochs (including validation) took ~20 seconds each.
600 epochs would be ~3.3 hours if throughput persists; this excludes subsequent
physical evaluation. Endpoint-only sampling has a 16.67 THz Nyquist limit.
The physical-evaluation configuration records two 100 ps NVE and a 15 ns NVT
production target, but the launched service performs training and independent
endpoint testing ONLY. Trajectory generation/analysis have not yet been run or
claimed complete for this network.

GPU compute benchmark on L40S / batch 8 / FP32:
unfactored 28.41 ms vs factored 25.76 ms per forward/backward/AdamW step, with
peak allocated CUDA memory 1.281 GB vs 1.080 GB. See
`reports/relational_prototype/gpu_throughput.json`. This ~1.10x GPU improvement
must not be confused with the larger CPU improvement or a comparison against
the old trained SMDP model.

## Geometry-complete relational block (R2)

The `geometry` block keeps the candidate-endpoint feedback architecture but
replaces each legacy relational block with one unified interaction cycle:

1. Project node vectors once per atom, then form invariant pair correlations.
2. Read both current and candidate-future edge axes using the same descriptor
   families while retaining their distinct roles.
3. Update persistent edges, aggregate invariant/equivariant messages, and apply
   a node-local learned vector/scalar mixing operation.

Transverse pair correlation is evaluated as
`u_i.u_j - (u_i.n)(u_j.n)`, avoiding explicit edge-wise transverse vector
tensors, triplets, rank-2 hidden features and attention matrices. The model is
still O(3)-equivariant and retains per-system total momentum/COM drift. It does
not establish energy conservation.

The selected formal R2 configuration is 160 scalar / 40 vector / 80 edge
channels with four blocks (2 proposal + 2 refinement), **1,750,760 trainable
parameters**. On an otherwise idle L40S, batch-8 FP32 forward + loss + backward
+ AdamW benchmark measured 36.24 ms and 1.674 GB peak allocated memory. Batch 8
had higher measured sample throughput than 16 or 32. The same mechanism at the
old 128/32/64 width has 1,130,784 parameters and measured 29.24 ms / 1.339 GB.
These are compute-only measurements on resident training graphs, not accuracy
or end-to-end training results. Reports are in `reports/relational_geometry_prototype/`.

Seventeen checks cover both legacy and geometry paths, including O(3), PBC,
batch/permutation/momentum behavior, factorized output/gradient identity,
future-feedback gradients, and direct gradient access through both current and
candidate axes. R1 was paused after epoch 353 without deleting its best/last
weights; R2 trains from scratch in a separate directory.
