# DSRT Water 5 fs formal training

Status: running from the frozen R3-Opt architecture on 2026-11-03.

## Protocol

- System: 64 flexible water molecules (192 atoms), 5 fs transition.
- Sources: three independent train NVE trajectories, one validation trajectory,
  and one test trajectory at 0.5 fs source resolution.
- Samples: 5,000 train / 1,250 validation / 1,000 test.
- Model: `geometry_gated`, 192 scalar / 48 vector / 96 edge channels,
  2 proposal + 2 refinement blocks, 2,357,616 parameters.
- Training: FP32, batch 8, 250,000 optimizer updates, 5,000-step warmup,
  cosine decay from 3e-4 to 1e-5, proposal weight 0.02.
- Selection: validation endpoint loss only; test remains unopened until model
  selection completes.

All 7,250 endpoint pairs were checked against their raw NVE HDF5 sources before
training. The largest stored-displacement discrepancy was 1.49e-8 Å and graph
endpoint reconstruction error was 8.88e-16 Å. Source trajectory sets are
disjoint across splits.

## Server paths

- Run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/water5_20261103_r3_opt`
- Log: `.../water5_20261103_r3_opt/train.log`
- Dataset cache: `.../data/smdp_reference_md_v1/water/derived_graphs/dsrt_5fs`
- Service: `dsrt-water5-r3opt-20261103.service`

The first epoch is a warmup point, not a selected result: validation position
RMSE 0.024251 Å and velocity RMSE 0.0067493 Å/fs at optimizer step 625.
