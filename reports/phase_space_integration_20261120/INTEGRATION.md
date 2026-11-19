# Phase-space operator: trainer integration and measured cost

2026-11-20. Branch `research/phase-space-operator-20261120`. This report covers
engineering integration and cost measurement only. **No formal training was
launched, no test split was read, no checkpoint was selected and no MD or
physical metric was produced.**

## Delivered

- `train_relational.py` now dispatches on `config["architecture"]`:
  `phase_space_operator_v0` builds `PhaseSpaceTransition`; absent or `dsrt`
  keeps the historical `RelationalTransition`. An unknown architecture raises.
- TRAIN-only normalization for the new model: position/velocity target RMS and
  `aggregation_scale = sqrt(mean_atoms sum_edges cosine_cutoff^2)`, computed once
  over the complete 5,000-graph TRAIN split and frozen as model buffers.
- The frozen scales are written into each checkpoint config under
  `normalization`, so `relational_inference.load_relational` reconstructs the
  exact model without touching data.
- Model-specific gradient diagnostics groups and block-activation hooks
  (node/edge RMS) are wired into the existing diagnostics path.
- Launch-format configs: `configs/phase_space_water5_20261120.json` and
  `configs/phase_space_quartz30_20261120.json`. Both still carry
  `"formal_training_started": false`.
- `scripts/benchmark_phase_space.py` measures real TRAIN-batch optimization cost
  and one-graph model-only inference on the same device.
- `tests/test_phase_space_integration.py`: normalization definition, trainer
  dispatch, one full backward step, gradient groups, diagnostics, checkpoint
  round-trip through the inference loader.

Full test suite: **77 passed** (14 phase-space tests).

## Measured cost

Device: one L40S, FP32, TF32 disabled. Batch 8 fixed TRAIN graphs (indices 0-7;
the design trace used a different eight-graph sample, hence slightly different
edge counts). AdamW `lr=0`, AMSGrad, gradient clip 10. Timing excludes dataset
I/O and neighbor-list rebuild; graphs are resident on device. Warmup 3, timed
iterations 10, median.

| System | Model | Params | Atoms / edges | Train step | Peak CUDA | One-graph forward |
|---|---|---:|---:|---:|---:|---:|
| Water 5 fs | DSRT R3-Opt (6 Å) | 2,357,616 | 1,536 / 138,220 | 0.1671 s | 6.99 GB | 7.54 ms |
| Water 5 fs | phase-space v0 | 3,036,224 | 1,536 / 138,220 | 0.1918 s | 4.39 GB | 14.46 ms |
| Quartz 30 fs | DSRT R3-Opt (4.5 Å) | 2,357,616 | 1,296 / 37,150 | 0.0422 s | 1.99 GB | 5.53 ms |
| Quartz 30 fs | phase-space v0 | 3,036,224 | 1,296 / 37,150 | 0.0697 s | 1.59 GB | 14.59 ms |

Relative to the frozen DSRT model on the same device:

| System | Train step | One-graph forward | Peak training memory |
|---|---:|---:|---:|
| Water 5 fs | +14.8% | 1.92x | -37.2% |
| Quartz 30 fs | +65.1% | 2.64x | -20.3% |

Model-only upper bounds from these step times (neighbor rebuild and I/O
excluded): Quartz 468 -> 178 ns/day; Water 57 -> 30 ns/day. These are **not**
end-to-end simulation rates and must not replace a rollout measurement.

## Interpretation

The new operator reduces peak training memory but is not cheaper in wall time.
The wider 128-channel node state and node-local CG products dominate the narrow
edge transport, especially at one-graph batch size where kernel launch and
small-tensor overhead matter. A five-step profiler trace (profiler overhead
included, so not comparable to the table; record
`quartz30_profile_l40s_20261120.json`) attributes the largest share (25.6%) to
`aten::bmm` and a further ~26% to GEMMs inside the node tensor-product path,
not to edge scatter. The predicted gain from moving capacity off edges is
therefore not realized by this unfused PyTorch reference implementation.

This changes the engineering premise, not the scientific question. A formal
250k-update run is affordable in memory; its cost is roughly 1.15x (Water) to
1.65x (Quartz) the historical training step, with roughly 2x slower autonomous
model calls. Any decision to train should account for that cost and for the
separate question of whether representation quality improves.

## Limits

- Shared L40S with 55-58% external utilization during measurement. Old and new
  models were timed back-to-back on the same device, but absolute times are not
  a dedicated-node benchmark.
- Single measurement run, 10 iterations, batch 8 only. No variance across
  devices or batch sizes.
- The implementation still materializes narrow edge messages and uses PyTorch
  scatter. No Triton fusion, no `torch.compile`, no mixed precision.
- Graph rebuild, rollout loop, analysis, NVT thermostat and physical metrics are
  outside this measurement.
- Only random initialization; no optimization or accuracy claim.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107 \
  /data1/wyd/envs/trajcast-c52f107/bin/python scripts/benchmark_phase_space.py \
  --system water5 --device cuda --batch-size 8 --iterations 10 \
  --output reports/phase_space_integration_20261120/water5_l40s_20261120.json
```

Raw records: `water5_l40s_20261120.json`, `quartz30_l40s_20261120.json`.
Profiler record and reproduce command:
`quartz30_profile_l40s_20261120.json`;

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107 \
  /data1/wyd/envs/trajcast-c52f107/bin/python scripts/profile_phase_space.py \
  --system quartz30 --steps 5 \
  --output reports/phase_space_integration_20261120/quartz30_profile_l40s_20261120.json
```
