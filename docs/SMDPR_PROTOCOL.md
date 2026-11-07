# SMDP-R: frozen first experiment, 2026-11-07

This user-authorized experiment supersedes the older DSRT-only restriction in
the parent AGENTS.md. Historical code, reports and weights remain unchanged.
New branch: `experiment/smdpr-relational-motion-modes`, based on the stable
geometry branch at `2043b80`.

## Fixed architecture

Use the complete R3-Opt `geometry_gated` 192/48/96, 2+2 network. Three invariant
32-dimensional node mode states persist through all four blocks. Every block
reuses its existing edge message and geometric invariants, aggregates distinct
mode-conditioned neighbor messages, updates the mode states, applies local
cross-mode coupling and scales the original scalar messages and vector
coefficients. Multiplicative scales are `1 + 0.1*tanh(condition)`. They are
nonzero at initialization and differentiable. No additional force call,
geometry graph, time integrator or independent expert GNN is present.

Disabling modes bypasses their entire path and reproduces the original network
with identical backbone weights, including floating-point output. Original
checkpoint keys and strict baseline loading are preserved. No old checkpoint
initializes the new primary models.

Parameters: original 2,357,616; full SMDP-R 2,571,592 (+9.08%), including the
23,376-parameter query head. Baseline modules are initialized first so the same
seed retains the same backbone initialization.

The query head reads detached final scalar/vector/mode states and detached
predicted endpoints. A Hermite boundary interpolant plus learned equivariant
interior corrections preserves both endpoint positions and endpoint velocities.
All corrections vanish at endpoints. Its independent optimizer cannot update
the endpoint backbone or modes. Its q/v outputs are independent: no exact
derivative identity or energy conservation is claimed. Query consistency and
segment joins must be measured.

## Stage 1

Two primary runs: Quartz 30 fs and flexible Water 5 fs. Use every original
5,000 train start and 1,250 validation start, retaining the independent
1,000-sample test split. All query targets are actual reference frames.
Existing `derived_segments/cmis_*fs` HDF5 files are reusable **data-only**
caches; no CMIS architecture or training code is imported. The loader checks
every sample's species, masses, initial q/v and endpoint against the existing
graph cache. Quartz samples 3 interior times from 3,6,...,27 fs; Water samples
3 from 1,2,3,4 fs, without interpolation. No additional reference MD is run.

FP32, TF32 disabled; batch 8; AdamW AMSGrad, weight decay 0; gradient clip 10;
250,000 optimizer updates, 400 complete epochs. Learning rate is linearly
warmed from 1e-5 to 3e-4 over 5,000 updates, then cosine-decayed to 1e-5. The
objective is the original `0.5*(normalized q MSE + normalized v MSE)` plus
0.02 times the same proposal objective. Scales use training data only.
Seeds match previous runs: Quartz 20261102, Water 20261103. The separately
optimized query reader uses the same learning-rate schedule, with a separate
sampling RNG; it does not alter the endpoint objective or shuffling RNG.
Best weights are chosen by validation endpoint loss only; no plateau stopping.

Log complete component losses, steps, actual time, memory, query loss and
periodic modal affinity, attention, modulation and gradient statistics. Save
best/last with optimizer/scheduler and all RNG states. Tests are not loaded by
the trainer. Failed or stopped runs remain distinguishable from complete ones.

## Predeclared subsequent protocol

Before stage 2, evaluate held-out validation endpoint and 2/4/8-step errors,
then 10 ps NVE from the validation source trajectory. Stop the candidate's
long-trajectory pipeline if q or v endpoint RMSE exceeds 1.5 times the frozen
R3-Opt validation value, or if 10 ps fails the existing trajectory safety
rules. This is a protective gate, not checkpoint selection. Preserve failures.

For candidates passing the gate, stage 2 is 25,000 optimizer updates of exactly
two differentiable autonomous steps from original training starts that have
two true reference endpoints. Graphs rebuild from the model's predicted state.
The objective is the mean of the original endpoint-plus-proposal loss over
the two steps. AdamW AMSGrad, batch 8, FP32, clip 10; cosine LR 3e-5 to 1e-5,
no additional warmup. Choose by the corresponding two-step validation loss.
The query reader remains gradient-isolated. Apply exactly the same stage-2
budget, sources, objective and validation to the frozen R3-Opt control; record
its historical stage-1 budget, rather than claiming that budget was identical.
Do not generate new MD or force/energy labels.

A K=1 control keeps 96 total latent components (dimension 96) and the same
32-wide interaction bottleneck, original backbone, data and stage-1/stage-2
budgets. Report actual parameters and timings: equality is approximate, not a
claim of identical FLOPs. It has 2,647,368 parameters, 2.95% more than K=3.
Its single-mode attention is trivially constant. Each system's K=1 run is
queued after its K=3 protected evaluation pipeline; it is not running yet.

After validation-only checkpoint selection, separately report stage-1 and
stage-2 independent endpoint/query/2/4/8-step tests. Run 100 ps NVE and the
existing CSVR NVT protocols only for candidates passing stability gates.
Quartz NVT remains 1.5 ns equilibration + 15 ns production; Water remains
0.25 ns + 2.5 ns. All physics uses the existing BKS/SPC-Fw offline analysis.
Do not change thermostat or metric definitions to rescue failures.

## Efficiency and interpretation

The pretraining paired Blackwell benchmark measured baseline 5.80 ms and
SMDP-R 8.65 ms per prebuilt 162-atom graph, ratio 1.491. Thirty dense queries
including the endpoint forward took 9.37 ms. These are initialized-network
model timings, **not** end-to-end trajectory throughput or accuracy. An earlier
unoptimized implementation measured ratio 1.536; that measurement is retained.
All L40S cards were occupied at launch; L40S-specific acceptance and production
throughput remain pending. Primary trainings use the available Blackwell GPU
and record actual sharing. Do not extrapolate ns/day or exclusive L40S speed.

Water success requires motion-energy partition, bond/angle statistics, VACF,
VDOS, RDF, MSD/diffusion and thermostat exchange, not just endpoint accuracy.
Quaternion or molecule labels are not network inputs. Mode affinity differences
alone do not prove meaningful physical modes. Report mode removal effects,
K=1, matched R3-Opt fine tuning, and all failures before attributing benefits.

## Reproduce

Server runs live under `experiments/smdpr/{quartz30,water5}_20261107_k3`.
Configurations are `configs/smdpr_quartz30_k3.json` and
`configs/smdpr_water5_k3.json`. Run with a free GPU UUID:

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> bash scripts/run_smdpr.sh <absolute-run-directory>
# After interruption: append --resume; existing runs are never overwritten.
```

Source data and full checkpoints/logs stay on the server. Commit compact
machine-readable metrics and reports; never commit HDF5 trajectories or weights.
