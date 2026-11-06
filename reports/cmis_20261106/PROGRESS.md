# CMIS formal experiment progress

Snapshot: 2026-11-06T17:46:30.024583+08:00

Training uses the full formal split. These are intermediate validation
metrics, not selected final test results or MD physical evidence.

| System | Epoch / 350 | Optimizer steps | Best epoch | Val endpoint q RMSE (Å) | Val endpoint v RMSE (Å/fs) |
|---|---:|---:|---:|---:|---:|
| quartz30 | 15 | 9375 | 14 | 0.0106446 | 0.0014610 |
| water5 | 14 | 8750 | 13 | 0.0061510 | 0.0026171 |

The fixed training budget is 350 epochs (218,750 optimizer updates per
system). Both models start from random initialization. The current config
contains 431,827 trainable parameters. Formal short/multistep, modal
intervention and NVE/NVT assessments are queued after training. A physical
failure will be recorded with its actual completed duration.
