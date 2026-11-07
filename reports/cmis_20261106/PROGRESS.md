# CMIS formal experiment progress

Snapshot: 2026-11-07T11:03:36.266834+08:00

Training uses the full formal split. Training and final independent tests
are complete; detailed physical failures are retained in RESULTS.md.

| System | Epoch / 350 | Optimizer steps | Best epoch | Val endpoint q RMSE (Å) | Val endpoint v RMSE (Å/fs) |
|---|---:|---:|---:|---:|---:|
| quartz30 | 350 | 218750 | 347 | 0.0053180 | 0.0007048 |
| water5 | 350 | 218750 | 346 | 0.0013363 | 0.0006048 |

The fixed training budget is 350 epochs (218,750 optimizer updates per
system). Both models start from random initialization. The current config
contains 431,827 trainable parameters. Independent short/multistep and
modal assessments are complete. Quartz NVE/NVT and Water NVT fail their
safety criteria; there are no complete NVT production results. See
RESULTS.md and results.json for measured outcomes and missing analyses.
