# SMDP-R: implementation and official run status

Date: 2026-11-07. Branch: `experiment/smdpr-relational-motion-modes`.

**This is a running experiment, not a completed accuracy or physical report.**
The primary Quartz/Water models are training from random initialization.
Validation has not yet selected a final checkpoint; no new independent-test,
100 ps NVE or production NVT result is claimed.

## Implemented and checked

- Retain original 192/48/96 geometry-gated backbone, 2 proposal + 2 refinement.
- Three 32-dimensional invariant motion states propagate on the original
  neighbor graph and persist across both stages. Bounded conditioning changes
  scalar messages and equivariant vector coefficients inside all four blocks.
- Local cross-mode exchange feeds subsequent propagation, rather than only a
  final mixture. Mode affinity and gradients are recorded per block.
- Original endpoint readout, PBC and per-system mass-weighted COM remain.
- Detached endpoint-anchored q/v reader, real interior supervision and a
  separate optimizer. Queries cannot update the endpoint backbone or modes.
- Predeclared two-step fine tuning, matched R3-Opt control, K=1 control,
  validation safety gates and independent prediction/physics pipeline.
- Optional streamed 1 fs queries in autonomous rollout without repeated GNN
  encoding. Query kinematics and segment joins are explicitly recorded. Dense
  Water reference VDOS downsamples the actual 0.5 fs reference to 1 fs.

45 mathematical/gradient/regression checks passed, including baseline bitwise
identity when disabled; O(3), translation, boost, PBC, permutation, variable
size/batch independence, momentum, all-mode initial gradients, whole-path mode
removal, detached query gradients, exact query endpoints and differentiable
graph rebuilding across two steps. These checks did not fit a small model.

Parameter counts: R3-Opt 2,357,616; K=3 2,571,592 (+9.08%, including 23,376
query parameters); K=1 with 96 total latent channels 2,647,368. K=1 is an
approximate capacity control, not a claim of equal FLOPs. It has the same
official data/update budget and its actual runtime will be measured.

## Data adaptation correction

The first full-data preflight stopped before any optimizer update because
the graph cache wraps initial positions while segment HDF5 labels are unwrapped.
The loader now verifies that the difference is an integer lattice shift and
carries each atom's same shift across every label in its segment. All sample
species, masses, initial velocities and endpoints are checked against the
original graph cache. Real displacements are never minimum-imaged. Initial
failed startup tracebacks remain in the server logs. This fix did not change
the network, split, targets or physical protocol.

The old CMIS-named HDF5 files are reused strictly as caches of true reference
frames. No CMIS model, basis, solver or objective is imported.

## Current runs and logs

Root: `/data1/wyd/research/foundation_mlip_adaptation/experiments/smdpr`.

| System | Run directory | Service | Stage-1 updates |
|---|---|---|---:|
| Quartz 30 fs | `quartz30_20261107_k3` | `smdpr-quartz30-k3-20261107` | 250,000 |
| Water 5 fs | `water5_20261107_k3` | `smdpr-water5-k3-20261107` | 250,000 |

Each directory contains `train.log`, `progress.json`, `history.jsonl`,
`status.json`, `metadata.json`, `config.json`, `best.pt` and `last.pt` once
the first epoch is complete. Best/last include optimizers, schedules and RNG.

Both trainings currently share GPU 0, RTX PRO 6000 Blackwell 96 GB, UUID
`GPU-f48b2c84-f983-64e3-000b-d669b9e64454`; the L40S devices were occupied.
No unrelated training process was stopped. User service lingering was enabled
so the runs survive logout. Service stdout/stderr append to the server logs.
CMIS optimization was not restarted; historical artifacts remain untouched.

Follow-up services `smdpr-{quartz30,water5}-followup-20261107` wait for a true
completed primary run, then execute the guarded pipeline. Their output is
`followup.log`. K=1 directories `quartz30_20261107_k1` and
`water5_20261107_k1` are **queued, not trained**. They follow their system's
primary protected pipeline. They do not start if its short validation gate fails.
The stage-2 R3-Opt control has identical fine-tuning budget; its historical
stage-1 budget is explicitly retained, not described as matched.

## Performance evidence and remaining work

Blackwell paired, prebuilt 162-atom graph: baseline 5.80 ms, K=3 endpoint
8.65 ms (1.491x); forward plus thirty dense queries 9.37 ms. See
`benchmark_blackwell_pretraining.json`. An earlier slower implementation
(1.536x) is retained in `benchmark_blackwell_initial.json`. Reusing the sender
projection and batching query COM reductions improved implementation overhead
without changing the architecture or objective, before any training update.

These are **initialized-model timings**, excluding neighbor rebuild, thermostat
and output. They establish neither production throughput nor L40S acceptance.
Those measurements, final accuracy and physical comparisons are pending.
Larger replicated cells are used only for computational scaling benchmarks;
they are not independent large-system physical-generalization evidence.

`training_snapshot.json` and `training_current.png` contain the actual interim
history at export time. No final physics numbers are inserted into this report.

## Reproduce / inspect

```bash
tail -f /data1/wyd/research/foundation_mlip_adaptation/experiments/smdpr/quartz30_20261107_k3/train.log
tail -f /data1/wyd/research/foundation_mlip_adaptation/experiments/smdpr/water5_20261107_k3/train.log

# Fixed protocol, in the existing environment with the upstream PYTHONPATH:
CUDA_VISIBLE_DEVICES=<free-GPU-UUID> bash scripts/run_smdpr.sh <run-directory>
python -m smdpr.pipeline --run <completed-run-directory> --baseline-control
python -m smdpr.export --root /data1/wyd/research/foundation_mlip_adaptation/experiments/smdpr --output reports/smdpr
```

See [the frozen protocol](../../docs/SMDPR_PROTOCOL.md) for exact training,
selection, gating and fine-tuning values. Server HDF5, checkpoints and raw logs
are not committed. Missing and failed results remain explicit.
