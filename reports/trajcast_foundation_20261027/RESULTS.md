# TrajCast foundation — observed results

Pinned upstream, environment, protocol and limits: [audit](../../trajcast_based/AUDIT.md).
Large artifacts remain at `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_based/quartz_20261027`. Historical experiments are unchanged.

No adapter/segment training has been launched. Original16 configuration is not a trained baseline.

## Official 30 fs checkpoint / official 1000-sample test

| Metric | Measured | Paper Table S1 |
|---|---:|---:|
| Displacement MAE / Å | 0.0014124012 | 0.0014124 |
| Velocity MAE / Å/fs | 0.000113700814 | 0.0001137 |

## official30_100ps

Trajectory completed: False; physics completed: False.

## official30_15ns

Trajectory completed: False; physics completed: False.

100 ps is a preliminary physical validation, not 15 ns reproduction.
GPU sharing and host load affect wall time; no uncontended speedup claim is made.
