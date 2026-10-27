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

Trajectory completed: True; physics completed: True.

Temperature: 302.090 K.
VDOS overlap to published MD: {'8': 0.9485163422849896, '14': 0.9632765508817769}.
Published TrajCast overlap (source CSV): {'8': 0.9537981981519278, '14': 0.9658434337132454}.
See energy_conventions.json for raw/DSF-corrected means and absolute differences; centered similarity alone is insufficient.

## official30_15ns

Trajectory completed: False; physics completed: False.

100 ps is a preliminary physical validation, not 15 ns reproduction.
GPU sharing and host load affect wall time; no uncontended speedup claim is made.
