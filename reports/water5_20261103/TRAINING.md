# DSRT Water 5 fs formal training

Status: formal training and independent endpoint test completed on 2026-11-04.

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

## Final result

The complete 250,000-update schedule ran to epoch 400. Validation loss continued
to improve through the final low-learning-rate phase, so epoch 400 is the
validation-selected checkpoint. The independent test set was accessed only
after selection.

| Metric | Validation | Independent test |
|---|---:|---:|
| Position RMSE (Å) | 0.000657063 | 0.000654127 |
| Velocity RMSE (Å/fs) | 0.000238463 | 0.000237643 |
| Position MAE (Å) | 0.000455865 | 0.000453784 |
| Velocity MAE (Å/fs) | 0.000165762 | 0.000165311 |

Training used 22,601.4 seconds (6.28 hours), including the resumed first epoch,
and reached a peak allocated model memory of about 3.1 GB. The result establishes
accurate supervised 5 fs endpoints but does not by itself establish autonomous
liquid dynamics. Multistep and 100 ps NVE evaluation are complete and reveal
substantial NVE heating; see `RESULTS.md` and `results.json` for the results.
