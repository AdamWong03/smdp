# Water 5 fs: complete 2.5 ns NVT results

The fixed validation-selected epoch-400 checkpoint completed 250 ps
thermalization plus 2.5 ns production (550,000 neural steps). No forces
were used during neural propagation. This is one autonomous run.

## Main result

Numerical completion did not produce credible liquid dynamics. Molecular
translation and rotation are strongly suppressed, internal motion dominates,
RDF peaks over-sharpen, velocity spectra disagree, and molecular MSD
approaches a plateau. The original CSVR thermostat does not repair this
imbalance. All failed and unfavorable results are retained.

| Observable | DSRT | Reference MD |
|---|---:|---:|
| Temperature, 2,000 matched frames (K) | 311.645 | 300.203 |
| Molecular translation KE (eV) | 0.152413 | 2.459377 |
| Molecular rotation KE (eV) | 0.212012 | 2.502091 |
| Molecular internal KE (eV) | 7.329671 | 2.450151 |
| Total KE (eV) | 7.694095 | 7.411619 |
| Potential energy mean (eV) | -29.740230 | -27.681713 |
| Potential energy std (eV) | 0.669385 | 0.636370 |
| O–O RDF first peak height | 11.46100 | 3.16480 |
| O–H RDF first peak height | 16.22110 | 29.88259 |
| H–H RDF first peak height | 4.75652 | 2.73391 |
| H VDOS overlap | 0.214574 | Compared with reference |
| O VDOS overlap | 0.233060 | Compared with reference |

The full 500,000-step production mean temperature after thermostat is
311.4266 K (pre-thermostat 312.6341 K).
This full-step estimate is distinct from the 2,000-frame thermodynamic table.

## Thermostat balance

The network adds 0.02980992 eV kinetic energy per 5 fs step;
CSVR removes 0.02981182 eV per step.
CSVR remains at the reference 300 K target and 50 fs coupling time.
The stationary balance, including the sampled random innovation, predicts 311.4259 K,
matching the measured production mean. Maximum temperature-scaling error
is 4.09e-06 K. These measurements support a persistent
model kinetic-energy source with finite thermostat relaxation rather than
a changed target temperature or a demonstrated CSVR formula error.

## Diffusion and its limits

The preregistered analysis uses eight time blocks, molecular COM MSD,
50 fs sampling and the 10–100 ps fit window. It yields:

| Quantity | DSRT | Reference |
|---|---:|---:|
| Mean fitted slope / 6 (m²/s) | 2.89316e-13 | 1.88297e-09 |

The model MSD plateaus; several later blocks have poor linear fits or
negative slopes. Its tiny fitted value is therefore an effective slope,
not a well-established positive equilibrium diffusion coefficient. The
reference has a clear diffusive regime. No fitting window was adjusted.

## Execution and software repair

End-to-end throughput is 37.534 ns/day on a shared L40S,
with 6330.3 seconds of trajectory generation.
It includes graph updates, CSVR and output, but excludes offline physical
analysis; it is not an exclusive-GPU timing comparison.

The initial energy task failed during reference verification with a
`Bond atom missing in image check` error. Long unwrapped coordinates are
now supplied as wrapped coordinates plus consistent image flags. Molecular
translations across multiple periodic boxes preserve energy and forces.
The long reference has coordinates every 0.5 fs, thermo every 5 fs and no
saved forces, unlike the NVE files. Verification now matches MD steps
and checks energy where forces are unavailable; NVE force checks remain.
Eight long-reference energy checks differ by at most 0.00058 eV. The
original failure traceback remains on the server. The repaired task reuses
the completed trajectory and all completed analysis tasks.

## Artifacts

- Server trajectory: `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/water5_20261103_r3_opt/physics/nvt_replica_01/trajectory.h5`
- Server analysis: `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/water5_20261103_r3_opt/physics/nvt_replica_01/analysis`
- Compact metrics: `nvt_results.json`
- Curves and numeric samples: `nvt_*.png` and `nvt_*.csv`
- Protocol: `configs/water5_physics.json`
- Reproduce/resume: `bash scripts/run_water5_nvt.sh` (existing trajectory is reused).
