# TrajCast foundation — observed results

Pinned upstream, environment, protocol and limits: [audit](../../trajcast_based/AUDIT.md).
Large artifacts remain at `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_based/quartz_20261027`. Historical experiments are unchanged.

No adapter/segment training has been launched. Original16 configuration is not a trained baseline.

## Official 30 fs checkpoint / official 1000-sample test

| Metric | Measured | Paper Table S1 |
|---|---:|---:|
| Displacement MAE / Å | 0.0014124012 | 0.0014124 |
| Velocity MAE / Å/fs | 0.000113700814 | 0.0001137 |

## Independent 16 fs dataset — official loader roundtrip

| Split | Samples | Maximum absolute field difference |
|---|---:|---:|
| train | 5000 | 0.0 |
| val | 1250 | 0.0 |
| test | 1000 | 0.0 |

No cross-graph neighbor edges. Normalization uses training samples only.

## official30_100ps

Trajectory completed: True; physics completed: True.

Temperature: 302.090 K.
VDOS overlap to published MD: {'8': 0.9485163422849896, '14': 0.9632765508817769}.
Published TrajCast overlap (source CSV): {'8': 0.9537981981519278, '14': 0.9658434337132454}.
See energy_conventions.json for raw/DSF-corrected means and absolute differences; centered similarity alone is insufficient.

## official30_15ns

Trajectory completed: False; physics completed: False.

Execution status at publication: interrupted.

The previous foreground job is no longer alive. The log stops at step 2336/550000 during thermalization without a Python traceback; the termination cause is not established. This is not evidence of a model stability failure. The partial trajectory is preserved unchanged. A new persistent-service run starts from the original seed and initial state, not from this partial trajectory.

## official30_15ns_service1

Trajectory completed: True; physics completed: True.

Execution status at publication: completed.

Temperature: 301.215 K.
VDOS overlap to published MD: {'8': 0.9551794323260134, '14': 0.9667777519260481}.
Published TrajCast overlap (source CSV): {'8': 0.9537981981519278, '14': 0.9658434337132454}.
See energy_conventions.json for raw/DSF-corrected means and absolute differences; centered similarity alone is insufficient.

## Outstanding physical-protocol limitation

The 100 ps reconstructed Carré-convention potential-energy mean is about 7.63 eV below the published reference mean, even after removing the known DSF self-energy constant. This difference is unresolved; it must not be dismissed as an arbitrary zero-point shift. The original Quartz LAMMPS input deck is not present in the pinned upstream release.

100 ps is a preliminary physical validation, not 15 ns reproduction.
GPU sharing and host load affect wall time; no uncontended speedup claim is made.
