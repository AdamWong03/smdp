# SMDP-R interim results, 2026-11-08

These are real completed predictions and trajectories, **not a complete
experiment**. Both primary 250,000-update trainings and both 25,000-update
two-step fine-tuning runs finished. K=1 and matched R3-Opt fine-tuning controls
have not started. No causal benefit is attributed to modes without them.

## Independent test, 1,000 original samples each

| System | Frozen stage | Position RMSE (Å) | Velocity RMSE (Å/fs) |
|---|---|---:|---:|
| Quartz 30 fs | Endpoint training | 0.001732317 | 0.000111900 |
| Quartz 30 fs | Two-step fine tuning | 0.001721913 | 0.000113175 |
| Water 5 fs | Endpoint training | 0.000657691 | 0.000240003 |
| Water 5 fs | Two-step fine tuning | 0.000636243 | 0.000231823 |

Each checkpoint was chosen on its declared validation objective. These test
values do not select a checkpoint or alter the architecture/training protocol.

## Quartz: serious NVE dissipation despite good endpoint errors

Both two-step-model NVE trajectories completed 99.99 ps without a numerical
safety stop. Their physical results are nevertheless poor:

| Quantity | NVE 1 | NVE 2 |
|---|---:|---:|
| Model mean temperature (K) | 133.184 | 131.763 |
| Corresponding reference mean temperature (K) | 288.403 | 303.142 |
| Model total-energy change (eV) | -6.99843 | -8.23553 |
| Corresponding reference total-energy change (eV) | +0.00782 | +0.00194 |
| Model fitted drift (meV/atom/ps) | -0.20028 | -0.24653 |

Both kinetic and potential behavior must be considered; this is not merely
a temperature statistic or an exchange of fixed total energy between them.
The 10 ps numerical gate was deliberately broad and did not establish physical
fidelity. It must not be retrospectively described as a physical-success gate.

## Quartz NVT: trajectory completed, dense spectrum still pending

The two-step model completed 1.5 ns equilibration plus all 15 ns production,
550,000 macro steps. Existing completed analysis gives:

| Quantity | Model | BKS reference |
|---|---:|---:|
| Mean temperature (K) | 296.8215 | 299.4479 |
| Mean raw BKS potential energy (eV) | -3135.84546 | -3135.77631 |
| Potential-energy standard deviation (eV) | 0.398654 | 0.396605 |
| Si–O first RDF peak height | 36.8445 | 34.8293 |
| O–O first RDF peak height | 9.06653 | 9.18141 |

RDF RMSE: Si–O 0.19095, O–O 0.01373. O/Si VDOS overlap on **30 fs sampled
states only**: 0.95044/0.96380, limited to the common Nyquist band of
16.6667 THz. The 1 fs query VDOS analysis was interrupted and is being resumed
from its spectral cache; no dense overlap is claimed. Near-target NVT
temperature and low-frequency overlap do not negate the NVE dissipation.

Recorded complete-production throughput was 86.02 ns/day on shared RTX PRO
6000 Blackwell hardware, including streamed dense output. This is not an
exclusive-L40S result or a same-hardware speed comparison against old reports.

## Water: NVE trajectories and analysis completed; NVT running

Both two-step-model 100 ps NVE trajectories were generated. The first uses
test_05 frame 0; the second uses frame 50,000 (25 ps), from the same independent
test source, **not a second independent reference trajectory**.

For the first trajectory's 0–50 ps common reference interval:

| Quantity | Model | Reference |
|---|---:|---:|
| Mean temperature (K) | 256.105 | 275.834 |
| Molecular translational KE (eV) | 2.06181 | 2.26005 |
| Molecular rotational KE (eV) | 1.09274 | 2.29741 |
| Molecular internal KE (eV) | 3.16835 | 2.25251 |

Thus energy partition is still wrong. The first candidate's full 100 ps total
energy changes by -1.60127 eV; the reference covers only 50 ps, so no 100 ps
reference difference is invented. After the reference-alignment repair, the
second NVE analysis completed and the 2.5 ns NVT pipeline restarted. The long
NVT is not yet complete.

## Recovery of demonstrated software/execution failures

The second Water reference-alignment copy treated every top-level HDF5 item
as a dataset. The actual source contains a `thermo` group, producing
`AttributeError: 'Group' object has no attribute 'shape'`. The new helper copies
groups recursively, crops independently sampled thermo arrays by actual MD
step, preserves raw energies/forces/species, shifts time origins consistently
and publishes only a completed file atomically. Interrupted intermediate copies
are retained with an `.interrupted-*` suffix. A regression test covers this
error, separate thermo cadence, scalar datasets and safe reuse.

The user service manager also failed with systemd `Result=signal`; its processes
were no longer running at inspection. The signal's cause is not established.
Both follow-up pipelines have been restarted as detached processes, with
per-run locks and `followup.pid` files. Existing completed checkpoints,
trajectories, analysis tasks and spectral blocks are reused. No trajectory,
model, loss, thermostat or evaluation definition was changed for recovery.

Recovery logs live in each primary run's `followup_recovery_20261108.log`.
The complete report still requires dense Quartz spectra, remaining Water
analysis/NVT, the matched R3-Opt control and K=1 control. Negative results above
are retained regardless of their later outcomes.
