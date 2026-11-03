# R3-Opt autonomous dynamics progress

Checkpoint: validation-selected epoch 364 (`best.pt`, SHA256
`d59c8110762e328c8c59debf60820090b6ab912fc896e675db619af02e395508`).
No BKS force is called during propagation. BKS is used only for offline energy
evaluation.

## Independent autoregressive prediction

999 of the 1,000 independent test starts have a complete 240 fs reference
continuation. Neighbor lists are rebuilt from each predicted state.

| Autonomous time | Position RMSE (A) | Velocity RMSE (A/fs) | Temperature RMSE (K) |
|---:|---:|---:|---:|
| 30 fs | 0.00180043 | 0.000122079 | 1.099 |
| 60 fs | 0.00400217 | 0.000191593 | 1.909 |
| 120 fs | 0.00884047 | 0.000304786 | 3.042 |
| 240 fs | 0.01899178 | 0.000508462 | 5.058 |

## Two independent 100 ps NVE trajectories

Both trajectories completed 3,333 autonomous 30 fs steps without a safety
failure, non-finite state or collision.

| Initial state | Mean T, DSRT / reference (K) | Min distance (A) | Total energy change, DSRT / reference (eV) | Drift, DSRT / reference (meV atom^-1 ps^-1) | O / Si VDOS overlap |
|---|---:|---:|---:|---:|---:|
| test_09 | 298.64 / 288.40 | 1.483 | 1.302 / 0.0078 | 0.0579 / 0.00055 | 0.9543 / 0.9555 |
| test_10 | 304.57 / 303.14 | 1.477 | 0.689 / 0.0019 | 0.0886 / -0.00047 | 0.9549 / 0.9557 |

The VDOS comparison is restricted to the common 30 fs sampling bandwidth
(Nyquist 16.67 THz); it is not evidence for recovery of higher-frequency
quartz motion. O--O RDF RMSE is 0.0158/0.0247 for the two replicas. Si--O RDF
RMSE is 0.2736/0.2018; its main peak position is correct, but the peak height is
lower than reference by 2.98/2.01.

The NVE result is substantially more stable than earlier SMDP trajectories,
but it is not energy conserving at reference-MD accuracy. Endpoint inference
with the production Vesin periodic neighbor list is numerically equivalent to
the general AtomicGraph path to FP32 rounding and reaches about 291 ns/day in a
compute-only 100-step measurement. The original general graph path measured
39.2 ns/day end to end for the saved NVE trajectories.

## NVT status

A 1.5 ns thermalization plus 15 ns production NVT trajectory is currently
running with the frozen 300 K, 100 fs CSVR protocol. No NVT statistic is
reported here before production completes.
