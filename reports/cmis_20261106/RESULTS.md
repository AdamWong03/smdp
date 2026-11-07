# CMIS: first formal Quartz / Water results

Upstream model commit d7e20ed; formal integration cdde3bb. Each system
used 350 complete epochs, 218,750 optimizer updates, all 5,000 training
and 1,250 validation samples, seed 20261106, FP32 and batch 8.
Both services finished their prescribed attempt sequences. That does not
mean the requested physical trajectories completed: all NVT attempts
failed before production. Checkpoints and trajectories remain on server.

## Independent endpoint test (1,000 samples)

| System / model | Position RMSE (Å) | Velocity RMSE (Å/fs) |
|---|---:|---:|
| quartz30 CMIS | 0.005415515 | 0.000718411 |
| quartz30 old DSRT R3-Opt | 0.001800432 | 0.000122078 |
| water5 CMIS | 0.001319127 | 0.000599985 |
| water5 old DSRT R3-Opt | 0.000654127 | 0.000237643 |

CMIS endpoint errors are higher on both systems. Architectures, parameter
counts, supervision targets and training schedules differ; this is an
outcome comparison, not an isolated mechanism or matched-budget claim.
CMIS has 431,827 parameters versus 2,357,616 for DSRT R3-Opt.

## Selection and training cost

| System | Selected epoch | Training hours | Val endpoint q RMSE (Å) | Val endpoint v RMSE (Å/fs) |
|---|---:|---:|---:|---:|
| quartz30 | 347 | 3.905 | 0.00531803 | 0.00070479 |
| water5 | 346 | 4.334 | 0.00133626 | 0.00060480 |

## Multistep independent test

999 starts per system have enough remaining real reference frames for
all eight autonomous steps; no reference states are injected.

| System | Time (fs) | Position RMSE (Å) | Velocity RMSE (Å/fs) |
|---|---:|---:|---:|
| quartz30 | 30 | 0.00541530 | 0.00071838 |
| quartz30 | 60 | 0.01337770 | 0.00096190 |
| quartz30 | 120 | 0.02996614 | 0.00154103 |
| quartz30 | 240 | 0.06259236 | 0.00244475 |
| water5 | 5 | 0.00131913 | 0.00060000 |
| water5 | 10 | 0.00413362 | 0.00076715 |
| water5 | 20 | 0.00918182 | 0.00104432 |
| water5 | 40 | 0.01603218 | 0.00177478 |

## Autonomous physical outcomes

| System / initial state | Completed valid time (ps) | Completed request? | Outcome |
|---|---:|---|---|
| quartz30 / nve_test_09 | 0.660 | no | RuntimeError: physical safety stop: min=0.39326679706573486, preT=2682.398030276146 |
| quartz30 / nve_test_10 | 0.480 | no | RuntimeError: physical safety stop: min=0.5473340153694153, preT=1859.678381946535 |
| quartz30 / nvt_replica_01 | 0.690 | no | RuntimeError: physical safety stop: min=0.1752435266971588, preT=1962.2808629269548 |
| quartz30 / validation_nve | 0.630 | no | RuntimeError: physical safety stop: min=0.7504101991653442, preT=5293.095905039733 |
| water5 / nve_test_05 | 100.000 | yes | No numerical safety stop |
| water5 / nvt_replica_01 | 113.265 | no | RuntimeError: physical safety stop: min=0.5737346410751343, preT=485.2167467216222 |
| water5 / validation_nve | 50.000 | yes | No numerical safety stop |

### Quartz

Both independent test NVE attempts fail within 1 ps: 0.66 and 0.48 ps
of valid states. The NVT attempt fails after 0.69 ps during the requested
1.5 ns thermalization, leaving zero production steps. Close contacts and/or
excess temperature trigger the unchanged safety conditions. No full-length
Quartz NVE/NVT, equilibrium RDF or equilibrium VDOS is claimed.
Short failed-trajectory energy curves remain available in the per-run folders.

### Water

The independent NVE trajectory completes 100 ps. Comparisons with the
reference are restricted to its available 50 ps. The matched sampled
mean temperature is 188.038 K versus 275.834 K.
Total-energy change is -5.32182 eV versus -0.02349 eV.
Thus the newer model dissipates energy rather than reproducing reference
NVE behavior. Its molecular kinetic partition is less distorted than the
old internally overheated DSRT run, but the system is too cold overall.

| Kinetic component, common 50 ps | CMIS (eV) | Reference (eV) |
|---|---:|---:|
| translation_eV | 1.44743 | 2.26005 |
| rotation_eV | 1.54487 | 2.29741 |
| internal_eV | 1.65012 | 2.25251 |
| total_KE_eV | 4.64242 | 6.80997 |

Water H/O macro-state VDOS overlaps are 0.68487/0.88269.
The corresponding real-reference 1 fs versus model queried-1 fs overlaps are 0.68150/0.88238.
These improve over the older heated DSRT NVE, but neither the differing
temperature nor nonstationarity permits an equilibrium-fidelity claim.

The Water NVT attempt fails after 113.265 ps during the requested 250 ps
thermalization, with a rejected-step minimum distance 0.5737 Å. There is
no 2.5 ns production trajectory, diffusion coefficient or NVT distribution.
No thermostat parameters or failure criteria were adjusted.

### Incomplete validation analysis

Water validation NVE completes 50 ps, but its offline energy verification
fails the existing reference-accuracy threshold. Other analyses completed.
The failed verification record and traceback are retained. This task is
explicitly incomplete; no energy result is substituted or fabricated.

## Mode and segment diagnostics

| System | Endpoint contribution effective rank (4 modes) | Max qdot–v discrepancy (Å/fs) |
|---|---:|---:|
| quartz30 | 1.4646 | 3.38e-09 |
| water5 | 1.7600 | 3.73e-09 |

Individual mode deletion changes predictions, but modes remain strongly
correlated (and some Quartz contributions are anticorrelated). The effective
ranks do not support four independent physical motion modes. Attention,
frequency distributions, deletion effects and dense-query errors are included
in the full validation/test JSON files. Analytic kinematic consistency and
linear momentum handling do not guarantee accurate autonomous dynamics.
Separate versus dense queries agree up to FP32 rounding, not bitwise.

## Efficiency and limits

The complete Water 100 ps NVE run reaches 22.44 ns/day, including two graph
builds, neural updates, dense 1 fs output and file I/O on a shared L40S.
Short failed-run throughput is not treated as long-production speed.
Exclusive-GPU speedup over old methods is not established.

## Conclusion

The first full CMIS experiments are negative overall: supervised accuracy
is worse, Quartz propagation fails early and neither NVT production
completes. Water NVE motion partition and spectrum improve in some respects
but exhibit substantial cooling and energy loss. No model or training
configuration was changed in response to these results.

## Reproduction and artifacts

- Exact training/data/evaluation settings: `PROTOCOL.md` and `configs/cmis_*_20261106.json`.
- Full measured metrics: `results.json`; per-split evaluations: `quartz30/` and `water5/`.
- Curves and tables for each successful/failed physical attempt are retained below those folders.
- Server run root: `/data1/wyd/research/foundation_mlip_adaptation/experiments/cmis`.
- Weights, raw histories and full HDF5 trajectories stay on the server.
- Re-export: `python scripts/export_cmis_results.py --base /path/to/experiments/cmis --out reports/cmis_20261106`.
- The failed NVT requests were not rerun or reported as complete.
