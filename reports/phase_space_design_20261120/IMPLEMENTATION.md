# Operator prototype: implementation status

2026-11-20. Branch `research/phase-space-operator-20261120`, based on audit commit
`237f7c7`. This is **design implementation**, not a trained method or a completed
simulation experiment. The old model, trainer, checkpoints and physical results
were not changed.

## Delivered

- [Complete calculation and training specification](../../paper/PHASE_SPACE_OPERATOR_DESIGN_20261120.md).
- [Independent operator prototype](../../phase_space_transition.py), 3,036,224 parameters.
- [Algebra/gradient tests](../../tests/test_phase_space_transition.py).
- [Executable TRAIN trace](../../scripts/check_phase_space_design.py).
- [Raw observations](implementation_checks.json), including frozen full-TRAIN
  normalization, actual tensor shapes, CG paths and parameter groups.
- Explicit design configurations under [configs/design](../../configs/design/).

## Verification actually performed

`tests/test_phase_space_transition.py`: **9 passed**. Coverage includes proper
rotations and reflections; translation/velocity boosts; unequal-size batch
independence and atom permutation; COM/momentum; PBC reimaging and graph rebuild;
checkpoint roundtrip; final-loss gradients through candidate geometry; nonzero
second-order-to-vector and receiver-motion paths; zero-speed finite derivatives;
FP32 backward; cutoff edge-removal limit; removal of both future condition paths
under the diagnostic `current_only` switch.

Tests use smaller channel widths to check identities; separate checks execute
the **full 128/32/64/4-width model** on eight real TRAIN graphs per system.

| Full-width input | Quartz 30 fs | Water 5 fs |
|---|---:|---:|
| Cutoff / Å | 4.5 | 6.0 |
| Normalization source samples | 5,000 TRAIN | 5,000 TRAIN |
| Forward/backward batch graphs | 8 | 8 |
| Batch atoms | 1,296 | 1,536 |
| Batch edges | 37,240 | 137,898 |
| q target RMS / Å | 0.0710899356 | 0.0560818662 |
| v target RMS / Å/fs | 0.0036142698 | 0.0128717573 |
| Fixed aggregation scale | 1.42889730 | 2.73474883 |
| All parameter gradients finite | yes | yes |
| Optimizer updates | 0 | 0 |

No validation/test states, old model weights, external potential, MD labels,
autonomous trajectory, optimizer or model selection were used. The reported
random-initialization objective in JSON is only a finite-computation check.
It is not an accuracy result and was not used to select the architecture.

## Cost evidence and limits

The implementation keeps `128x0e+128x1o+128x2e` on nodes and only
`32x0e+32x1o+32x2e` in transported messages. Each of four blocks has 679,200
parameters, of which 592,512 belong to node-local coupling. There are 11 allowed
CG order paths, with channelwise products and learned surrounding channel maps.

For the first Water graph (17,276 edges), the individual FP32 geometry tensor
is 4.745 MiB, edge concatenation is 11.401 MiB, and gate tensor is 10.544 MiB.
These are single tensor sizes, **not peak-memory measurements**. The actual graph
also contains angular bases, saved autograd tensors and wider node features.

The JSON records a single cold CPU forward/backward with two threads. These
times include cold execution effects and are not a comparative benchmark.
No GPU throughput, memory peak, acceleration ratio or ns/day is claimed.
The current implementation still materializes narrow edge messages and uses
PyTorch scatter; it is not a fused sparse kernel implementation.

## Reproduce

No new environment was installed. The existing isolated environment contains
PyTorch `2.5.1+cu121`, e3nn `0.5.9` and the pinned TrajCast data container.
The new network does not import a TrajCast model. The project's other environment
and its top-level requirements were not modified.

From the worktree:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=.:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107 \
/data1/wyd/envs/trajcast-c52f107/bin/python -m pytest \
  tests/test_phase_space_transition.py -q --disable-warnings
```

The full-width trace refuses to overwrite its output:

```bash
SMDP_RESEARCH_ROOT=/data1/wyd/research/foundation_mlip_adaptation \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data1/wyd/envs/trajcast-c52f107/bin/python \
  scripts/check_phase_space_design.py \
  --output /tmp/phase_space_design_recheck.json
```

This reads only the preserved `quartz30_train;P:64.pt` and `water5_train;P:64.pt`
caches recorded in JSON. No cache is regenerated. Source code was independently
written; no unlicensed E2Former kernel or copied upstream model was added.

## Remaining before formal training

The production trainer and loader still dispatch the historical models. They
must explicitly construct this prototype, save/load its TRAIN normalization
buffers, validate graph/horizon contracts at input boundaries, and record its
own diagnostic parameter groups. This work is not silently represented as done.

Measure same-device forward/backward and simulation-step cost with synchronized
timing before committing an expensive formal run. Use the existing train splits,
endpoint objective and planned optimization budget when integrating. Do not
replace this prerequisite with a new architecture search or external reproduction.

Scientific unknowns remain: the effectiveness of the message bottleneck,
phase-space representation, aggregation scale and node coupling; optimization
behavior; short-term accuracy; long-term physical fidelity. Algebraic validity
does not establish any of these.
