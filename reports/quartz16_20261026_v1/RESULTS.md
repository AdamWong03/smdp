# Neural segment experiment — observed status

Method/configuration: [frozen protocol](../../neural_segment/README.md). No absent result is considered completed.

Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/neural_segment/quartz16_20261026_v1`. Raw logs, weights and trajectories remain there.

| Model | Run | Actual ps | Trajectory complete | Physics complete | Mean T bias / K | BKS end-start / eV |
|---|---|---:|---|---|---:|---:|
| direct | nve_screen_val_07 | 10.0 | True | True | 69.11179290888924 | 5.329733529483747 |
| direct | nve_screen_val_08 | 10.0 | True | True | 74.03162499678353 | 5.244019485603076 |
| independent | nve_screen_val_07 | 10.0 | True | True | 157.21125527214178 | 9.273939442467508 |
| independent | nve_screen_val_08 | 10.0 | True | True | 222.76239635471853 | 13.353628049680992 |
| full | nve_screen_val_07 | 10.0 | True | True | 294.0595241928702 | 14.261965355236043 |
| full | nve_screen_val_08 | 10.0 | True | True | 394.0990415679365 | 20.14456796760396 |

Compare matched run lengths and cadence only. See metrics JSON for VDOS, RDF and sampled full energy curves; a thermostat is not evidence of energy conservation.
