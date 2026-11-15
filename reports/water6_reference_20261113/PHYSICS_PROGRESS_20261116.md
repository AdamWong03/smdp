# First 20 ps validation NVE: endpoint improvement is insufficient

At approximately 01:20 CST, 2026-11-16. Frozen DSRT 6 Å epoch 398, validation
source val_04/frame 0, 4,000 autonomous 5 fs steps. The complete requested
trajectory was generated without numerical failure. No thermostat or reference
forces were used during propagation. No new checkpoint was selected.

## Motion statistics (0–20 ps)

| Molecular KE component, eV | DSRT 6 Å | Same-window reference MD |
|---|---:|---:|
| Translation | 1.36561 | 2.21525 |
| Rotation | 1.53773 | 2.23611 |
| Internal | 4.41816 | 2.33788 |
| Total | 7.32150 | 6.78924 |

Mean temperature is 296.55 K versus reference 274.99 K. These are nonstationary
20 ps NVE window averages, not independent equilibrium replicates. A temperature
near 300 K does not compensate for incorrect molecular motion partition.
The smaller endpoint error has not removed the systematic imbalance in this
validation rollout. Comparison against historical 4.5 Å must wait for the
same-initial-state evaluation, now resumed below.

## Physical analysis is incomplete

Motion, RDF, VDOS and diffusion tasks produced outputs. The strict energy task
failed its unchanged reference verification: one of eight fixed frames has
absolute PE error 0.001170599 eV, exceeding 0.001 eV. Force maximum error at that
frame is 0.004829 eV/Å, below the existing 0.02 eV/Å threshold. No tolerance,
potential parameter, energy offset or frame selection was changed.

`reference_coordinate_audit.json` compares the exact same eight frames using
saved float32 unwrapped positions versus float64 reconstruction from saved
wrapped positions plus image flags/cell. Differences reach about 5.1e-5 Å and
the reconstructed representation changes the energy discrepancies. This flags
coordinate representation/rounding sensitivity; it is not proof that every
frame meets the tolerance. The production analysis is not silently switched
to the representation that happens to pass these eight points.

The existing full-energy task only checked the first/last reference frames,
whereas the energy task checked eight. Its previously generated curve must
therefore **not** be treated as a fully verified energy result while the shared
eight-frame reference check fails. Original outputs and failure are retained
on the server. Energy drift is withheld from formal conclusions at this stage.

The shell pipeline previously stopped on this analysis failure before starting
multistep evaluation and the historical arm. It now preserves nonzero final
failure status while continuing those independent evaluations. No successful
trajectory is regenerated. Full native TrajCast training continues at epoch 61;
its current validation-selected best remains epoch 59. Native test and
autonomous physics have not been evaluated yet.

Next decisions require the matched 4.5 Å trajectory and native baseline physics;
this result does not authorize another architecture or changed loss.
