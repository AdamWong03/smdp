# R3-Opt Quartz 30 fs physical results

The validation-selected epoch-364 checkpoint was frozen before physical
evaluation. Propagation uses no BKS force or energy call. The trajectory uses
30 fs autonomous steps; all BKS energies below are offline evaluations.

## Short autonomous error

| Time | Position RMSE (A) | Velocity RMSE (A/fs) | Temperature RMSE (K) |
|---:|---:|---:|---:|
| 30 fs | 0.001800 | 0.0001221 | 1.10 |
| 60 fs | 0.004002 | 0.0001916 | 1.91 |
| 120 fs | 0.008840 | 0.0003048 | 3.04 |
| 240 fs | 0.018992 | 0.0005085 | 5.06 |

999 independent test starts had a complete 240 fs reference continuation.

## 100 ps NVE

Both independent trajectories completed without a collision, non-finite state
or safety stop.

| Initial state | Mean T, DSRT / BKS (K) | Minimum distance (A) | Total-energy change, DSRT / BKS (eV) | Drift, DSRT / BKS (meV atom^-1 ps^-1) | O / Si VDOS overlap |
|---|---:|---:|---:|---:|---:|
| test_09 | 298.64 / 288.40 | 1.483 | +1.302 / +0.0078 | 0.0579 / 0.00055 | 0.9543 / 0.9555 |
| test_10 | 304.57 / 303.14 | 1.477 | +0.689 / +0.0019 | 0.0886 / -0.00047 | 0.9549 / 0.9557 |

The trajectories are structurally stable and improve strongly over earlier
SMDP energy failures, but do not conserve the BKS reference energy to MD
accuracy.

## 15 ns NVT after 1.5 ns thermalization

The complete 550,000-step trajectory finished without failure. Production is
500,000 autonomous steps (15 ns).

| Observable | R3-Opt | BKS reference | Difference |
|---|---:|---:|---:|
| Temperature mean (K) | 300.228 | 299.448 | +0.780 |
| Temperature standard deviation (K) | 19.951 | 19.117 | +0.834 |
| Kinetic-energy mean (eV) | 6.2480 | 6.2318 | +0.0162 |
| O kinetic-energy mean (eV) | 4.2117 | 4.1584 | +0.0533 |
| Si kinetic-energy mean (eV) | 2.0363 | 2.0734 | -0.0370 |
| Potential-energy mean (eV) | -3135.6579 | -3135.7763 | +0.1184 |
| Potential-energy standard deviation (eV) | 0.4352 | 0.3966 | +0.0386 |
| Si--O RDF RMSE | 0.1799 | -- | -- |
| O--O RDF RMSE | 0.02405 | -- | -- |
| O VDOS overlap | 0.94647 | reference split-half: 0.99923 | -- |
| Si VDOS overlap | 0.94973 | reference split-half: 0.99903 | -- |

The Si--O and O--O main peak positions match reference. The Si--O peak height
is 32.95 versus 34.83 (5.4% low); the O--O peak height is 9.282 versus 9.181.
The minimum sampled distance is 1.432 A.

VDOS is evaluated only over the common 30 fs sampling bandwidth, with a
16.67 THz Nyquist limit. It does not claim recovery of inaccessible
higher-frequency quartz motion.

### Thermostat balance

The network changes kinetic energy by +0.000961 eV per 30 fs step on average;
CSVR removes -0.000961 eV per step. The predicted stationary post-thermostat
temperature from this balance is 300.132 K. Thus the accurate mean NVT
temperature is not produced by compensating a large monotonic heating or
cooling rate, although instantaneous exchanges remain broad.

## Efficiency

- Hardware: one NVIDIA L40S
- End-to-end 1.5 ns + 15 ns trajectory: 106.1 ns/day
- Runtime: 13,441.9 s
- Model parameters: 2,357,616
- Model peak allocated CUDA memory: about 58 MB
- Compute-only 100-step measurement with the production Vesin neighbor path:
  about 291 ns/day

The difference between compute-only and end-to-end throughput includes HDF5
streaming, thermostat work and per-step statistics.

## Independent 15 ns NVT repeat

A second trajectory used reference frame 7,500,000 and an independent CSVR
seed. It completed the same 1.5 ns thermalization and 15 ns production protocol
without a collision, non-finite value or safety stop.

| Observable | Replica 1 | Replica 2 | BKS reference |
|---|---:|---:|---:|
| Mean temperature (K) | 300.228 | 300.604 | 299.448 |
| Temperature standard deviation (K) | 19.951 | 19.184 | 19.117 |
| Mean potential energy (eV) | -3135.657945 | -3135.657887 | -3135.776305 |
| Potential-energy standard deviation (eV) | 0.43524 | 0.42997 | 0.39660 |
| Si--O RDF RMSE | 0.17993 | 0.17900 | -- |
| O--O RDF RMSE | 0.02405 | 0.02227 | -- |
| O VDOS overlap | 0.94647 | 0.94665 | split-half: 0.99923 |
| Si VDOS overlap | 0.94973 | 0.95013 | split-half: 0.99903 |
| Minimum sampled distance (A) | 1.432 | 1.456 | -- |
| End-to-end throughput (ns/day) | 106.1 | 127.4 | -- |

The Si--O peak height is 32.90 in replica 2 versus 34.83 in the reference;
the O--O peak height is 9.255 versus 9.181. The network adds 0.000872 eV of
kinetic energy per step on average and CSVR removes 0.000872 eV. These agree in
direction and scale with replica 1 (+0.000961/-0.000961 eV).

The two independent runs therefore reproduce both the favorable results and
the limitations: temperature, O--O structure, potential-energy mean and
low-frequency spectra are highly repeatable, while the Si--O peak deficit,
element kinetic-energy imbalance and excess potential-energy variance persist.
Two replicas establish repeatability but are not treated as a complete
uncertainty estimate; the frozen protocol still calls for a third run.

## Research conclusion

R3-Opt produces a stable long NVT trajectory with strong low-frequency VDOS,
temperature, O--O structure and energy-distribution agreement. Remaining
errors are visible in NVE energy drift, Si--O peak height, element-resolved
kinetic-energy allocation and the potential-energy variance. These results
support freezing the architecture and extending physical conditions and system
sizes rather than changing the network again.
